"""Unit tests for the evolutionary idea search module.

All tests mock ModelRegistry.completion so no real LLM calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scider.agents.ideation_agent.idea_search import (
    IdeaNode,
    IdeaSearchResult,
    combine_ideas,
    improve_idea,
    run_idea_search,
    score_population,
)
from scider.workflows.paper_bootstrap import build_idea_from_ideation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_idea(title: str = "Test Idea", description: str = "A test description.") -> dict:
    return {
        "title": title,
        "description": description,
        "rationale": "Some rationale.",
        "experiment": "Run experiment X.",
        "contribution": "Expected contribution.",
        "novelty_score": 7.0,
    }


def _make_ranking_response(n: int, best_first: list[int] | None = None) -> str:
    """Return a valid ranking JSON string for n ideas."""
    ranking = best_first if best_first else list(range(1, n + 1))
    return json.dumps({"ranking": ranking, "brief_rationale": "Top idea is most novel."})


def _make_idea_response(title: str = "Improved Idea") -> str:
    return json.dumps({
        "title": title,
        "description": "An improved description.",
        "rationale": "Better rationale.",
        "experiment": "Better experiment.",
        "contribution": "Better contribution.",
    })


# ---------------------------------------------------------------------------
# score_population
# ---------------------------------------------------------------------------

class TestScorePopulation:
    def test_assigns_composite_scores_to_all_nodes(self):
        ideas = [_make_idea(f"Idea {i}") for i in range(3)]
        nodes = [IdeaNode(idea=idea) for idea in ideas]

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            msg.content = _make_ranking_response(3)
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            score_population(nodes, "test query")

        for node in nodes:
            assert node.composite_score is not None
            assert 0.0 <= node.composite_score <= 1.0
            assert node.score_novelty is not None
            assert node.score_feasibility is not None
            assert node.score_impact is not None
            assert node.score_specificity is not None

    def test_neutral_score_on_parse_failure(self):
        nodes = [IdeaNode(idea=_make_idea()) for _ in range(2)]

        def bad_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            msg.content = "not json at all !!!"
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=bad_completion,
        ):
            score_population(nodes, "test query")

        # Should get neutral 0.5 for all dimensions, not raise
        for node in nodes:
            assert node.score_novelty == 0.5
            assert node.score_feasibility == 0.5
            assert node.score_impact == 0.5
            assert node.score_specificity == 0.5
            assert node.composite_score is not None

    def test_top_ranked_idea_gets_highest_score(self):
        nodes = [IdeaNode(idea=_make_idea(f"Idea {i}")) for i in range(3)]

        call_count = [0]

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            # Always rank idea 2 first (1-based), idea 1 second, idea 3 last
            msg.content = _make_ranking_response(3, best_first=[2, 1, 3])
            call_count[0] += 1
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            score_population(nodes, "test query")

        # Idea at index 1 (1-based id=2) should have higher novelty score than others
        assert nodes[1].score_novelty is not None
        assert nodes[1].score_novelty > nodes[0].score_novelty  # type: ignore[operator]
        assert nodes[1].score_novelty > nodes[2].score_novelty  # type: ignore[operator]


# ---------------------------------------------------------------------------
# improve_idea
# ---------------------------------------------------------------------------

class TestImproveIdea:
    def test_returns_dict_with_required_keys(self):
        idea = _make_idea()

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            msg.content = _make_idea_response("Improved Title")
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            result = improve_idea(idea, "specificity", "test query")

        assert "title" in result
        assert "description" in result
        assert "rationale" in result
        assert "experiment" in result
        assert "contribution" in result

    def test_returns_fallback_on_llm_failure(self):
        idea = _make_idea("Original")

        def bad_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            raise RuntimeError("LLM down")

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=bad_completion,
        ):
            result = improve_idea(idea, "novelty", "test query")

        assert result["title"] == "Original"

    def test_preserves_original_novelty_score(self):
        idea = _make_idea()
        idea["novelty_score"] = 8.5

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            msg.content = _make_idea_response()
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            result = improve_idea(idea, "feasibility", "test query")

        assert result["novelty_score"] == 8.5


# ---------------------------------------------------------------------------
# combine_ideas
# ---------------------------------------------------------------------------

class TestCombineIdeas:
    def test_returns_dict_with_required_keys(self):
        idea_a = _make_idea("Idea A")
        idea_b = _make_idea("Idea B")

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            msg.content = _make_idea_response("Combined Idea")
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            result = combine_ideas(idea_a, idea_b, "test query")

        assert "title" in result
        assert "description" in result

    def test_returns_fallback_on_llm_failure(self):
        idea_a = _make_idea("Idea A")
        idea_b = _make_idea("Idea B")

        def bad_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            raise RuntimeError("LLM down")

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=bad_completion,
        ):
            result = combine_ideas(idea_a, idea_b, "test query")

        assert result["title"] == "Idea A"  # fallback to idea_a


# ---------------------------------------------------------------------------
# run_idea_search
# ---------------------------------------------------------------------------

class TestRunIdeaSearch:
    def _make_mocked_completion(self, n_ideas: int):
        """Return a fake completion that always answers ranking for n_ideas."""
        def fake(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            prompt = messages[0].content if messages else ""
            msg = MagicMock()
            if "Rank" in (system_prompt or ""):
                # Scoring call
                msg.content = _make_ranking_response(n_ideas)
            else:
                # Operator call
                msg.content = _make_idea_response("Generated Idea")
            return msg
        return fake

    def test_skips_search_with_single_idea(self):
        ideas = [_make_idea()]
        result = run_idea_search(ideas, "test query")

        assert result.best_ideas == ideas
        assert result.llm_calls_used == 0
        assert result.iterations_completed == 0
        assert not result.search_budget_hit

    def test_terminates_at_budget(self):
        ideas = [_make_idea(f"Idea {i}") for i in range(5)]

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            if "Rank" in (system_prompt or ""):
                msg.content = _make_ranking_response(len(ideas))
            else:
                msg.content = _make_idea_response()
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            # Set budget so low the initial scoring alone hits it
            result = run_idea_search(ideas, "test query", max_llm_calls=4)

        assert result.search_budget_hit
        assert result.llm_calls_used >= 4

    def test_returns_enriched_idea_dicts(self):
        ideas = [_make_idea(f"Idea {i}") for i in range(3)]

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            if "Rank" in (system_prompt or ""):
                msg.content = _make_ranking_response(3)
            else:
                msg.content = _make_idea_response()
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            result = run_idea_search(ideas, "test query", max_llm_calls=10)

        assert len(result.best_ideas) > 0
        top = result.best_ideas[0]
        assert "composite_score" in top
        assert "score_novelty" in top
        assert "search_operator" in top

    def test_sorted_by_composite_score_descending(self):
        ideas = [_make_idea(f"Idea {i}") for i in range(4)]

        def fake_completion(model_name, messages, system_prompt=None, agent_sender=None, **kwargs):
            msg = MagicMock()
            if "Rank" in (system_prompt or ""):
                msg.content = _make_ranking_response(len(ideas))
            else:
                msg.content = _make_idea_response()
            return msg

        with patch(
            "scider.agents.ideation_agent.idea_search.ModelRegistry.completion",
            side_effect=fake_completion,
        ):
            result = run_idea_search(ideas, "test query", max_llm_calls=8)

        composites = [i["composite_score"] for i in result.best_ideas if i.get("composite_score")]
        assert composites == sorted(composites, reverse=True)


# ---------------------------------------------------------------------------
# backward-compat: build_idea_from_ideation
# ---------------------------------------------------------------------------

class TestBuildIdeaFromIdeationBackwardCompat:
    def test_falls_back_to_novelty_score_without_composite(self):
        ideas = [
            {**_make_idea("Idea Low"), "novelty_score": 3.0},
            {**_make_idea("Idea High"), "novelty_score": 9.0},
        ]
        result = build_idea_from_ideation(ideas, None, "test query")
        assert "Idea High" in result

    def test_prefers_composite_score_over_novelty(self):
        ideas = [
            {**_make_idea("Idea High Novelty"), "novelty_score": 9.0, "composite_score": 0.3},
            {**_make_idea("Idea High Composite"), "novelty_score": 5.0, "composite_score": 0.9},
        ]
        result = build_idea_from_ideation(ideas, None, "test query")
        assert "Idea High Composite" in result

    def test_selected_index_overrides_scoring(self):
        ideas = [
            {**_make_idea("Idea First"), "composite_score": 0.2},
            {**_make_idea("Idea Second"), "composite_score": 0.9},
        ]
        result = build_idea_from_ideation(ideas, 0, "test query")
        assert "Idea First" in result

    def test_handles_empty_ideas(self):
        result = build_idea_from_ideation([], None, "my research query")
        assert "my research query" in result
