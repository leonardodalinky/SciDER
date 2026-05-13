"""Unit tests for the eval prompt monkey-patches.

We patch upstream's ``OpenAI`` client + ``get_response`` so no real
API call is made — we just verify that our replacement functions:
  1. correctly call the prompt-building path with the new prompts,
  2. parse responses the way upstream expects, and
  3. successfully install via ``apply_patches`` / are reverted by
     ``revert_patches``.

The bench's ``eval/`` directory has both ``eval.py`` (a module) and
``new_eval.py`` (the scorer) under the same dir name. To avoid the
sys.path conflict that prevents ``from eval.new_eval import ...``
from working in normal test runs, we prepend the upstream root to
sys.path manually here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_UPSTREAM_ROOT = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "discoverybench" / "discoverybench"
)
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

# Skip the whole module gracefully if upstream deps aren't installed —
# we don't want to fail CI on machines that lack langchain / openai / etc.
pytest.importorskip("openai")

from eval import new_eval  # noqa: E402
from eval import scider_eval_patches as patches  # noqa: E402

# --------------------------------------------------------------------------- #
# apply / revert wiring                                                       #
# --------------------------------------------------------------------------- #


class TestApplyRevert:
    def setup_method(self):
        # Make sure each test starts with stock upstream functions.
        patches.revert_patches()

    def teardown_method(self):
        patches.revert_patches()

    def test_apply_replaces_extractor_and_matcher(self):
        before_extr = new_eval.get_sub_hypotheses
        before_match = new_eval.match_context_with_gpt
        patched = patches.apply_patches()
        assert sorted(patched) == ["get_sub_hypotheses", "match_context_with_gpt"]
        assert new_eval.get_sub_hypotheses is patches.scider_get_sub_hypotheses
        assert new_eval.match_context_with_gpt is patches.scider_match_context_with_gpt
        assert new_eval.get_sub_hypotheses is not before_extr
        assert new_eval.match_context_with_gpt is not before_match

    def test_revert_restores_originals(self):
        # capture originals BEFORE apply
        orig_extr = new_eval.get_sub_hypotheses
        orig_match = new_eval.match_context_with_gpt
        patches.apply_patches()
        patches.revert_patches()
        assert new_eval.get_sub_hypotheses is orig_extr
        assert new_eval.match_context_with_gpt is orig_match

    def test_apply_is_idempotent(self):
        orig_extr = new_eval.get_sub_hypotheses
        patches.apply_patches()
        patches.apply_patches()  # 2nd call must not stack
        patches.revert_patches()
        assert new_eval.get_sub_hypotheses is orig_extr


# --------------------------------------------------------------------------- #
# Replacement extractor                                                       #
# --------------------------------------------------------------------------- #


_FAKE_METADATA = {
    "datasets": [
        {
            "name": "data.csv",
            "description": "test dataset",
            "columns": [
                {"name": "x", "description": "x col"},
                {"name": "y", "description": "y col"},
            ],
        }
    ]
}


class _FakeClient:
    """Trivial stand-in for ``openai.OpenAI()`` — we never actually call
    it because ``get_response`` is mocked, but the constructor must
    succeed without an API key."""

    pass


class TestExtractor:
    def test_prompt_carries_granularity_rule_and_examples(self):
        """The extractor prompt must include the abstract-scope rule and
        at least one of our calibration examples (we'd never want the
        few-shot examples silently dropped)."""
        captured: dict = {}

        def _fake_get_response(client, prompt, model=None, max_retry=1):
            captured["prompt"] = prompt
            return {
                "sub_hypo": [
                    {
                        "text": "h",
                        "context": "c",
                        "variables": ["x"],
                        "relations": "r",
                    }
                ]
            }

        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", _fake_get_response),
        ):
            out = patches.scider_get_sub_hypotheses(
                query="Q",
                hypo="agent hypo",
                workflow="agent workflow",
                dataset_meta=_FAKE_METADATA,
                llm_used="gpt-4o",
                dataset_type="synth",
                use_column_metadata=True,
            )
        prompt = captured["prompt"]
        # Hard rule from the audit.
        assert "most abstract scope phrase" in prompt
        # Examples must be there to calibrate.
        assert "ml1/ml3/rpp" in prompt
        # The agent's own text must be in the payload, untouched.
        assert "agent hypo" in prompt and "agent workflow" in prompt
        # Result shape preserved.
        assert out["full_hypo"] == "agent hypo"
        assert out["sub_hypo"][0]["context"] == "c"

    def test_handles_percent_character_in_hypothesis(self):
        """Upstream uses ``prompt % (a, b, c)`` which crashes on a literal
        ``%`` in any of the values. Our replacement must survive."""
        captured: dict = {}

        def _fake_get_response(client, prompt, model=None, max_retry=1):
            captured["prompt"] = prompt
            return {"sub_hypo": []}

        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", _fake_get_response),
        ):
            patches.scider_get_sub_hypotheses(
                query="Q",
                hypo="GDP grew by 5% in 2023",
                workflow="step 1: computed 50%% growth and 100% coverage",
                dataset_meta=_FAKE_METADATA,
                llm_used="gpt-4o",
                dataset_type="synth",
            )
        # Both literal % values must appear in the prompt verbatim
        # (we don't accidentally mangle them).
        assert "5%" in captured["prompt"]
        assert "100%" in captured["prompt"]

    def test_none_response_returns_empty_subhypo(self):
        """When the LLM call fails (None), upstream and our replacement
        both must return an empty sub_hypo list with full_hypo set."""
        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", lambda *a, **k: None),
        ):
            out = patches.scider_get_sub_hypotheses(
                query="Q",
                hypo="h",
                workflow="w",
                dataset_meta=_FAKE_METADATA,
                llm_used="gpt-4o",
                dataset_type="synth",
            )
        assert out == {"sub_hypo": [], "full_hypo": "h"}


# --------------------------------------------------------------------------- #
# Replacement matcher                                                         #
# --------------------------------------------------------------------------- #


class TestMatcher:
    def test_prompt_includes_subset_rule_and_examples(self):
        captured: dict = {}

        def _fake_get_response(client, prompt, model=None):
            captured["prompt"] = prompt
            return {"match": True, "rationale": "subset"}

        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", _fake_get_response),
        ):
            assert (
                patches.scider_match_context_with_gpt(
                    gold_hyp="h_g",
                    gold_context="In Psychology",
                    pred_hyp="h_p",
                    pred_context="In Psychology, specifically Social",
                    model="gpt-4o",
                )
                is True
            )
        p = captured["prompt"]
        # Subset rule must be explicit.
        assert "subset" in p
        # Calibration examples for the dominant failure pattern.
        assert "In Psychology" in p
        assert "Social and Cognitive disciplines" in p
        # No-match examples.
        assert "In Economics" in p

    def test_returns_false_when_response_is_not_dict(self):
        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", lambda *a, **k: None),
        ):
            assert (
                patches.scider_match_context_with_gpt("h", "ctx_g", "h", "ctx_p", model="gpt-4o")
                is False
            )

    def test_returns_match_value_from_dict(self):
        for v, expected in [
            (True, True),
            (False, False),
            ("true", True),
            ("", False),
            (1, True),
            (0, False),
        ]:
            with (
                patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
                patch.object(patches, "get_response", lambda *a, _v=v, **k: {"match": _v}),
            ):
                assert patches.scider_match_context_with_gpt(
                    "h", "g", "h", "p", model="gpt-4o"
                ) is bool(expected)


# --------------------------------------------------------------------------- #
# End-to-end integration with new_eval (via apply_patches)                    #
# --------------------------------------------------------------------------- #


class TestIntegration:
    def teardown_method(self):
        patches.revert_patches()

    def test_is_matching_context_uses_our_replacement(self):
        """``is_matching_context`` calls ``match_context_with_gpt`` by
        bare name — our monkey-patch must take effect even through that
        indirection."""
        patches.apply_patches()
        # Force the slow path (non-identical, non-None contexts) so it
        # actually invokes match_context_with_gpt.
        with (
            patch.object(patches, "OpenAI", lambda *a, **k: _FakeClient()),
            patch.object(patches, "get_response", lambda *a, **k: {"match": True}),
        ):
            result = new_eval.is_matching_context(
                gold_hyp="h",
                gold_context="In Psychology",
                pred_hyp="h",
                pred_context="In Psychology, sub-area",
                llm_used="gpt-4o",
            )
        assert result is True
