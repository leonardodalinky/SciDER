"""Tests for scider.agents.critic_review — verdict extraction."""

from scider.agents.critic_review import _extract_verdict


class TestExtractVerdict:
    def test_strong_pass(self):
        feedback = "### Overall Assessment\n**Strong** — excellent work."
        assert _extract_verdict(feedback) == "pass"

    def test_adequate_pass(self):
        feedback = "## Overall Assessment: Adequate\nSome minor issues."
        assert _extract_verdict(feedback) == "pass"

    def test_good_pass(self):
        feedback = "Overall Assessment\nGood work overall."
        assert _extract_verdict(feedback) == "pass"

    def test_needs_improvement_retry(self):
        feedback = "### Overall Assessment\n**Needs improvement** — critical gaps."
        assert _extract_verdict(feedback) == "retry"

    def test_poor_retry(self):
        feedback = "Overall Assessment: Poor\nMultiple critical issues."
        assert _extract_verdict(feedback) == "retry"

    def test_critical_issue_retry(self):
        feedback = "There is a critical issue with the data processing."
        assert _extract_verdict(feedback) == "retry"

    def test_critical_error_retry(self):
        feedback = "Found a critical error in the methodology."
        assert _extract_verdict(feedback) == "retry"

    def test_ambiguous_defaults_pass(self):
        feedback = "The analysis covered the main points. Some areas could be improved."
        assert _extract_verdict(feedback) == "pass"

    def test_empty_defaults_pass(self):
        assert _extract_verdict("") == "pass"
