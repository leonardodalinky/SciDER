"""Per-workflow API usage tracker.

Listens to every Message produced by the backend and accumulates token counts
and estimated cost per model role. Register with add_message_listener() and
retrieve results via get_rows() when the workflow finishes.
"""

from __future__ import annotations

import threading

from scider.core.types import Message


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}  # role -> {model, input_tokens, output_tokens, calls}

    def on_message(self, msg: Message) -> None:
        input_t = msg.prompt_tokens or 0
        output_t = msg.completion_tokens or 0
        if not msg.llm_sender or not (input_t or output_t):
            return
        role = msg.llm_sender
        # Some senders (e.g. the Claude Agent SDK coding subagent) report an
        # exact cost directly — that price correctly accounts for prompt-cache
        # read/write rates, so prefer it over a flat litellm token estimate.
        cost_override = getattr(msg, "cost_usd", None)
        with self._lock:
            if role not in self._stats:
                model_id = _model_id_for_role(role)
                self._stats[role] = {
                    "model": model_id,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    # tokens from messages WITHOUT an exact cost — estimated below
                    "est_input_tokens": 0,
                    "est_output_tokens": 0,
                    "cost_override_sum": 0.0,
                    "has_override": False,
                }
            s = self._stats[role]
            s["input_tokens"] += input_t
            s["output_tokens"] += output_t
            s["calls"] += 1
            if cost_override is not None:
                s["cost_override_sum"] += cost_override
                s["has_override"] = True
            else:
                s["est_input_tokens"] += input_t
                s["est_output_tokens"] += output_t

    def get_rows(self) -> list[dict]:
        """Return per-role stats sorted by total token usage (descending)."""
        with self._lock:
            rows = []
            for role, s in self._stats.items():
                est = _estimate_cost(s["model"], s["est_input_tokens"], s["est_output_tokens"])
                if s["has_override"]:
                    # Exact provider cost + estimate for any non-exact messages.
                    cost = s["cost_override_sum"] + (est or 0.0)
                else:
                    cost = est
                rows.append(
                    {
                        "role": role,
                        "model": s["model"],
                        "input_tokens": s["input_tokens"],
                        "output_tokens": s["output_tokens"],
                        "calls": s["calls"],
                        "cost_usd": cost,
                    }
                )
            return sorted(rows, key=lambda r: -(r["input_tokens"] + r["output_tokens"]))

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(s["input_tokens"] + s["output_tokens"] for s in self._stats.values())

    @property
    def total_cost(self) -> float | None:
        rows = self.get_rows()
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        return sum(costs) if costs else None


def _model_id_for_role(role: str) -> str:
    try:
        from scider.core.llms import ModelRegistry

        return ModelRegistry.instance().models.get(role, {}).get("model", "unknown")
    except Exception:
        return "unknown"


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model_id,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None
