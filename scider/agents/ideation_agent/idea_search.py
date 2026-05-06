"""Evolutionary idea search for the ideation agent.

Grounded in AIDE (arxiv 2502.13138), AIRA/AIRA-2 (arxiv 2507.02554 / 2603.26499),
and AI-Scientist v2 (arxiv 2504.08066).

Key adaptation from those systems (which use objective ML fitness): scores here are
K-way batch *rankings* per dimension, not absolute values, to reduce LLM scoring noise.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from scider.core.llms import ModelRegistry
from scider.core.types import Message
from scider.core.utils import parse_json_from_llm_response

LLM_NAME = "ideation"
AGENT_NAME = "ideation"

# Set > 0 when running on a rate-limited API tier (e.g. Gemini free 20 req/min).
# Adds this many seconds between sequential dimension-scoring calls.
# Recommended: 4.0 for Gemini free tier (keeps calls ~15/min with operator overhead).
_SEQUENTIAL_CALL_DELAY_S: float = 0.0

# Search hyperparameters
# POPULATION_SIZE=8, SURVIVORS=4 → n_new=4, n_improve=round(4*0.75)=3, n_combine=1
# Ensures Combine runs every iteration (would be 0 with POPULATION_SIZE=6).
POPULATION_SIZE = 8
MAX_ITERATIONS = 3
MAX_LLM_CALLS = 60
SURVIVORS_PER_ITER = 4
IMPROVE_FRACTION = 0.75
MIN_POPULATION_TO_SEARCH = 2

SCORE_WEIGHTS = {
    "novelty": 0.30,
    "feasibility": 0.25,
    "impact": 0.25,
    "specificity": 0.20,
}

DIMENSIONS = {
    "novelty": "Which idea opens the most genuinely new research territory?",
    "feasibility": "Which idea could a PhD student realistically run as their dissertation project?",
    "impact": "Which idea, if it succeeds, would most change the field or benefit the most people?",
    "specificity": "Which idea has the clearest path from hypothesis to a measured experimental result?",
}

IMPROVE_INSTRUCTIONS: dict[str, str] = {
    "novelty": (
        "Find a more unique angle, an unexplored combination of concepts, or a gap in the "
        "literature that this idea is best positioned to fill. The revised idea must differ "
        "substantively from what has been done — not just a reframing of the same contribution."
    ),
    "feasibility": (
        "Replace any speculative or resource-intensive components with tractable alternatives. "
        "The experiment must be completable in 12-18 months with standard research lab resources. "
        "Scope down where necessary; do not invent resources that do not exist."
    ),
    "impact": (
        "Broaden or sharpen the problem addressed. Consider: who else benefits if this succeeds, "
        "what downstream capabilities this enables, or whether a stronger problem framing increases "
        "the significance of a positive result."
    ),
    "specificity": (
        "Add a clear null hypothesis, at least one baseline comparison, a measurable primary "
        "outcome metric, and the key confounds to control. Replace vague phrases like "
        "'we will explore' with concrete procedures and expected measurements."
    ),
}

_RANKING_SYSTEM = (
    "You are an expert scientific research evaluator. "
    "Rank research ideas on a specific quality dimension. Be discriminating — do not give tied ranks."
)

_IMPROVE_SYSTEM = (
    "You are a research idea editor. Revise to strengthen one specific quality dimension "
    "while preserving the core identity of the idea."
)

_COMBINE_SYSTEM = (
    "You are a research idea synthesizer. Produce one new idea that combines the strongest "
    "elements of two parent ideas. The result must be genuinely synthetic — "
    "not just idea A with B's title."
)


@dataclass
class IdeaNode:
    idea: dict
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_ids: list[str] = field(default_factory=list)
    operator: str = "seed"  # "seed" | "improve" | "combine"
    score_novelty: Optional[float] = None
    score_feasibility: Optional[float] = None
    score_impact: Optional[float] = None
    score_specificity: Optional[float] = None
    composite_score: Optional[float] = None
    generation: int = 0
    # Top-dimension rationale from the ranker (used by Combine operator)
    top_rationale: dict[str, str] = field(default_factory=dict)

    def weak_dimension(self) -> Optional[str]:
        """Return the dimension with the lowest normalized rank score."""
        scores = {
            "novelty": self.score_novelty,
            "feasibility": self.score_feasibility,
            "impact": self.score_impact,
            "specificity": self.score_specificity,
        }
        valid = {k: v for k, v in scores.items() if v is not None}
        if not valid:
            return None
        return min(valid, key=lambda k: valid[k])

    def best_dimension(self) -> Optional[str]:
        scores = {
            "novelty": self.score_novelty,
            "feasibility": self.score_feasibility,
            "impact": self.score_impact,
            "specificity": self.score_specificity,
        }
        valid = {k: v for k, v in scores.items() if v is not None}
        if not valid:
            return None
        return max(valid, key=lambda k: valid[k])


@dataclass
class IdeaSearchResult:
    best_ideas: list[dict]
    all_nodes: list[IdeaNode]
    llm_calls_used: int
    iterations_completed: int
    search_budget_hit: bool
    initial_best_composite: float = 0.0  # best composite among seed population after first scoring


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _ranking_prompt(nodes: list[IdeaNode], dimension: str, user_query: str) -> str:
    lines = [
        f"Research topic: {user_query}",
        "",
        f'Rank the following {len(nodes)} research ideas by **{dimension}** — '
        f"{DIMENSIONS[dimension]}",
        "",
    ]
    for i, node in enumerate(nodes, 1):
        title = node.idea.get("title", f"Idea {i}")
        desc = node.idea.get("description", "")
        lines.append(f"IDEA {i}: {title}")
        if desc:
            lines.append(desc)
        lines.append("")
    lines += [
        'Return ONLY a JSON object with this exact format:',
        '{"ranking": [<1-based id of best>, <2nd best>, ..., <worst>], '
        '"brief_rationale": "<1-2 sentences on why the top-ranked idea leads on this dimension>"}',
        "",
        "Rules:",
        f'- "ranking" must be a permutation of [1, 2, ..., {len(nodes)}]. No ties.',
        "- Do not add explanation outside the JSON.",
    ]
    return "\n".join(lines)


def _score_dimension(nodes: list[IdeaNode], dimension: str, user_query: str) -> None:
    """Score all nodes on one dimension via K-way batch ranking. Mutates nodes in-place."""
    n = len(nodes)
    prompt = _ranking_prompt(nodes, dimension, user_query)
    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            [Message(role="user", content=prompt)],
            system_prompt=_RANKING_SYSTEM,
            agent_sender=AGENT_NAME,
            temperature=0.2,
        )
        data = parse_json_from_llm_response(msg.content or "")
        ranking: list[int] = data.get("ranking", [])
        rationale: str = data.get("brief_rationale", "")

        if len(ranking) != n or set(ranking) != set(range(1, n + 1)):
            raise ValueError(f"Invalid ranking {ranking} for n={n}")

        for node, rank_1based in zip(nodes, [ranking.index(i + 1) + 1 for i in range(n)]):
            normalized = (n + 1 - rank_1based) / n
            setattr(node, f"score_{dimension}", normalized)
            # Store rationale on the node that ranked first
            if rank_1based == 1 and rationale:
                node.top_rationale[dimension] = rationale

    except Exception as e:
        logger.warning("Scoring failed for dimension {}: {}. Assigning neutral 0.5.", dimension, e)
        # Unconditionally overwrite — a stale score from a previous population context
        # is worse than neutral: it mixes scales across different batch sizes.
        for node in nodes:
            setattr(node, f"score_{dimension}", 0.5)


def _compute_composite(node: IdeaNode) -> None:
    scores = {
        "novelty": node.score_novelty,
        "feasibility": node.score_feasibility,
        "impact": node.score_impact,
        "specificity": node.score_specificity,
    }
    if any(v is None for v in scores.values()):
        return
    node.composite_score = sum(SCORE_WEIGHTS[k] * scores[k] for k in SCORE_WEIGHTS)  # type: ignore[operator]


def score_population(nodes: list[IdeaNode], user_query: str, max_workers: int = 4) -> None:
    """Score all nodes across all 4 dimensions, then compute composite.

    Args:
        max_workers: Parallel scoring threads. Set to 1 on rate-limited API tiers
            (e.g. Gemini free tier with 20 req/min) to avoid quota errors.

    Note: when nodes include both survivors and new nodes, a survivor's composite_score
    may change as the ranking batch changes — this is correct behavior (calibration),
    not a bug.
    """
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_score_dimension, nodes, dim, user_query): dim
                for dim in DIMENSIONS
            }
            for future in as_completed(futures):
                dim = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Dimension {} scoring thread raised: {}", dim, e)
    else:
        for dim in DIMENSIONS:
            _score_dimension(nodes, dim, user_query)
            if _SEQUENTIAL_CALL_DELAY_S > 0:
                import time
                time.sleep(_SEQUENTIAL_CALL_DELAY_S)

    for node in nodes:
        _compute_composite(node)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _parse_idea_response(raw: str, fallback_idea: dict) -> dict:
    """Parse an operator's JSON response into an idea dict."""
    try:
        data = parse_json_from_llm_response(raw or "")
        required = {"title", "description"}
        if not required.issubset(data.keys()):
            raise ValueError(f"Missing required keys: {required - data.keys()}")
        return {
            "title": data.get("title", fallback_idea.get("title", "")),
            "description": data.get("description", fallback_idea.get("description", "")),
            "rationale": data.get("rationale", fallback_idea.get("rationale", "")),
            "experiment": data.get("experiment", fallback_idea.get("experiment", "")),
            "contribution": data.get("contribution", fallback_idea.get("contribution", "")),
            "novelty_score": fallback_idea.get("novelty_score"),  # preserve original
        }
    except Exception as e:
        logger.warning("Failed to parse operator response: {}. Returning fallback.", e)
        return fallback_idea


def improve_idea(idea: dict, weak_dim: str, user_query: str) -> dict:
    """Generate a targeted improvement of one idea on its weakest dimension."""
    instruction = IMPROVE_INSTRUCTIONS.get(weak_dim, "Improve the overall quality of this idea.")
    prompt = f"""Research topic: {user_query}

Original idea:
Title: {idea.get("title", "")}
Description: {idea.get("description", "")}
Rationale: {idea.get("rationale", "N/A")}
Experiment: {idea.get("experiment", "N/A")}
Contribution: {idea.get("contribution", "N/A")}

This idea scored weakest on: **{weak_dim}**

{instruction}

Produce a revised version. Return ONLY a JSON object with these exact keys:
{{"title": "...", "description": "2-3 sentences", "rationale": "...", "experiment": "...", "contribution": "..."}}"""

    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            [Message(role="user", content=prompt)],
            system_prompt=_IMPROVE_SYSTEM,
            agent_sender=AGENT_NAME,
            temperature=0.7,
        )
        return _parse_idea_response(msg.content or "", idea)
    except Exception as e:
        logger.warning("improve_idea failed: {}. Returning original.", e)
        return idea


def combine_ideas(
    idea_a: dict,
    idea_b: dict,
    user_query: str,
    rationale_a: str = "",
    rationale_b: str = "",
) -> dict:
    """Synthesize a new idea by combining the strongest aspects of two parents."""
    strongest_a = f"Strongest aspect: {rationale_a}" if rationale_a else ""
    strongest_b = f"Strongest aspect: {rationale_b}" if rationale_b else ""

    prompt = f"""Research topic: {user_query}

Idea A:
Title: {idea_a.get("title", "")}
Description: {idea_a.get("description", "")}
{strongest_a}

Idea B:
Title: {idea_b.get("title", "")}
Description: {idea_b.get("description", "")}
{strongest_b}

Synthesize a new research idea that:
1. Inherits the core mechanism or framing from one idea
2. Incorporates the novelty angle, experimental rigor, or domain scope from the other
3. Is more powerful than either parent alone

Return ONLY a JSON object with these exact keys:
{{"title": "...", "description": "2-3 sentences", "rationale": "...", "experiment": "...", "contribution": "..."}}

Do not describe the synthesis process. Return only the resulting idea."""

    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            [Message(role="user", content=prompt)],
            system_prompt=_COMBINE_SYSTEM,
            agent_sender=AGENT_NAME,
            temperature=0.8,
        )
        return _parse_idea_response(msg.content or "", idea_a)
    except Exception as e:
        logger.warning("combine_ideas failed: {}. Returning idea_a.", e)
        return idea_a


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------

def _rank_proportional_sample(nodes: list[IdeaNode], k: int) -> list[IdeaNode]:
    """Sample k nodes without replacement, weighted by composite_score rank."""
    import random

    if len(nodes) <= k:
        return list(nodes)

    # Assign rank weights: rank 1 (best) = N weight, rank N = 1 weight
    sorted_nodes = sorted(nodes, key=lambda n: n.composite_score or 0.0, reverse=True)
    n = len(sorted_nodes)
    weights = [n - i for i in range(n)]  # [n, n-1, ..., 1]

    selected = []
    remaining = list(zip(weights, sorted_nodes))
    for _ in range(k):
        total = sum(w for w, _ in remaining)
        r = random.random() * total
        cumulative = 0.0
        for idx, (w, node) in enumerate(remaining):
            cumulative += w
            if r <= cumulative:
                selected.append(node)
                remaining.pop(idx)
                break
    return selected


def run_idea_search(
    seed_ideas: list[dict],
    user_query: str,
    max_llm_calls: int = MAX_LLM_CALLS,
    score_workers: int = 4,
    max_iterations: int = MAX_ITERATIONS,
    improve_fraction: float = IMPROVE_FRACTION,
    _pre_scored_nodes: Optional[list["IdeaNode"]] = None,
) -> IdeaSearchResult:
    """Run evolutionary best-first search over the idea space.

    Args:
        seed_ideas: Initial ideas from the ideation agent (list of dicts).
        user_query: The original research topic, used in all operator prompts.
        max_llm_calls: Hard budget cap. Search terminates when this is reached.

    Returns:
        IdeaSearchResult with best_ideas sorted by composite_score descending.
    """
    if len(seed_ideas) < MIN_POPULATION_TO_SEARCH:
        logger.info("Too few ideas ({}) for search. Returning as-is.", len(seed_ideas))
        return IdeaSearchResult(
            best_ideas=seed_ideas,
            all_nodes=[],
            llm_calls_used=0,
            iterations_completed=0,
            search_budget_hit=False,
            initial_best_composite=0.0,
        )

    llm_calls = 0
    all_nodes: list[IdeaNode] = []
    iteration = 0

    if _pre_scored_nodes is not None:
        # Caller already scored the seeds (shared across ablations). Deep-copy so
        # this run's scoring mutations don't corrupt the caller's node objects.
        import copy
        population = copy.deepcopy(_pre_scored_nodes)
        all_nodes.extend(population)
    else:
        # Seed population — cap to POPULATION_SIZE
        population = [
            IdeaNode(idea=idea, operator="seed", generation=0)
            for idea in seed_ideas[:POPULATION_SIZE]
        ]
        logger.info("Idea search: scoring {} seed ideas across 4 dimensions.", len(population))
        score_population(population, user_query, max_workers=score_workers)
        llm_calls += 4
        all_nodes.extend(population)

    # Record best composite among seeds before any operators run.
    initial_best_composite = max(
        (n.composite_score or 0.0 for n in population), default=0.0
    )

    # Pin the best seed so it can never be displaced by iteration selection.
    # Without pinning, a high-quality seed can be eliminated when re-scored in a
    # larger population (n=8 vs n=6), even if it would beat evolved ideas on a
    # level playing field. The pinned seed always reaches the final scoring pass,
    # where it competes in the same n=8 context as all evolved ideas.
    pinned_seed = max(population, key=lambda n: n.composite_score or 0.0)

    any_new_nodes = False  # tracks whether operators ever produced new nodes

    # Main loop
    while iteration < max_iterations and llm_calls < max_llm_calls:
        iteration += 1
        logger.info("Idea search iteration {}/{}", iteration, max_iterations)

        # Select survivors: always keep the pinned seed; fill remaining slots
        # with the top scorers from the rest of the population.
        scored = [n for n in population if n.composite_score is not None]
        scored.sort(key=lambda n: n.composite_score or 0.0, reverse=True)
        rest = [n for n in scored if n is not pinned_seed]
        survivors = [pinned_seed] + rest[: SURVIVORS_PER_ITER - 1]

        n_new = POPULATION_SIZE - len(survivors)
        n_improve = round(n_new * improve_fraction)
        n_combine = n_new - n_improve

        new_nodes: list[IdeaNode] = []

        # Improve operator
        for i in range(n_improve):
            if llm_calls >= max_llm_calls:
                break
            parent = survivors[i % len(survivors)]
            weak_dim = parent.weak_dimension() or "specificity"
            logger.debug("improve_idea: parent='{}', weak_dim={}", parent.idea.get("title"), weak_dim)
            new_idea = improve_idea(parent.idea, weak_dim, user_query)
            llm_calls += 1
            new_nodes.append(
                IdeaNode(
                    idea=new_idea,
                    operator="improve",
                    parent_ids=[parent.node_id],
                    generation=iteration,
                )
            )

        # Combine operator
        for _ in range(n_combine):
            if llm_calls >= max_llm_calls:
                break
            if len(survivors) < 2:
                break
            a, b = _rank_proportional_sample(survivors, 2)
            rationale_a = a.top_rationale.get(a.best_dimension() or "", "")
            rationale_b = b.top_rationale.get(b.best_dimension() or "", "")
            logger.debug(
                "combine_ideas: '{}' x '{}'", a.idea.get("title"), b.idea.get("title")
            )
            new_idea = combine_ideas(a.idea, b.idea, user_query, rationale_a, rationale_b)
            llm_calls += 1
            new_nodes.append(
                IdeaNode(
                    idea=new_idea,
                    operator="combine",
                    parent_ids=[a.node_id, b.node_id],
                    generation=iteration,
                )
            )

        if new_nodes:
            any_new_nodes = True
            # Score against the full merged batch so ranks are calibrated.
            # A survivor's composite_score may shift — this is correct, not a bug.
            score_population(survivors + new_nodes, user_query, max_workers=score_workers)
            llm_calls += 4
            all_nodes.extend(new_nodes)

        population = survivors + new_nodes

    # Final authoritative scoring pass — only when the population changed.
    # Skipping when no operators ran keeps baseline_composite.lift = 0 by construction
    # and avoids paying 4 calls to re-score an unchanged population.
    if any_new_nodes:
        logger.info("Idea search: final scoring pass on {} ideas.", len(population))
        score_population(population, user_query, max_workers=score_workers)
        llm_calls += 4

    budget_hit = llm_calls >= max_llm_calls
    if budget_hit:
        logger.warning("Idea search hit LLM budget cap ({} calls).", llm_calls)

    best = sorted(population, key=lambda n: n.composite_score or 0.0, reverse=True)

    best_ideas: list[dict] = []
    for node in best:
        enriched = dict(node.idea)
        enriched["composite_score"] = node.composite_score
        enriched["score_novelty"] = node.score_novelty
        enriched["score_feasibility"] = node.score_feasibility
        enriched["score_impact"] = node.score_impact
        enriched["score_specificity"] = node.score_specificity
        enriched["search_operator"] = node.operator
        best_ideas.append(enriched)

    logger.info(
        "Idea search complete: {} iterations, {} LLM calls, best composite={:.3f}",
        iteration,
        llm_calls,
        best[0].composite_score if best else 0.0,
    )

    return IdeaSearchResult(
        best_ideas=best_ideas,
        all_nodes=all_nodes,
        llm_calls_used=llm_calls,
        iterations_completed=iteration,
        search_budget_hit=budget_hit,
        initial_best_composite=initial_best_composite,
    )
