"""Unit tests for the pure parts of the DiscoveryBench workflow.

We don't actually invoke the data/experiment agents here — that requires LLM
keys and significant compute. Instead we pin the glue logic that's most
likely to silently break: task discovery, prompt building, result.json
validation, scan-completed scanning, and ``run_one_task`` wiring (via a
patched ``run_full_workflow``).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bench_workflows import discoverybench_workflow as mod

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _write_metadata(
    path: Path,
    dataset: str,
    metadataid: int,
    queries: list[dict],
    columns: list[dict] | None = None,
    domain_knowledge: str | None = None,
    workflow_tags: list[str] | None = None,
) -> None:
    cols = columns or [
        {"name": "x", "description": "the x column"},
        {"name": "y", "description": "the y column"},
    ]
    meta = {
        "id": metadataid,
        "domain": dataset,
        "datasets": [
            {
                "name": "data.csv",
                "description": f"data for {dataset}",
                "columns": cols,
            }
        ],
        "queries": queries,
    }
    if domain_knowledge is not None:
        meta["domain_knowledge"] = domain_knowledge
    if workflow_tags is not None:
        meta["workflow_tags"] = workflow_tags
    path.write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def fake_bench_root(tmp_path: Path) -> Path:
    """Two real-test tasks, one with multiple metadata files + multiple queries."""
    bench_root = tmp_path / "bench"
    real_test = bench_root / "real" / "test"

    # Task 1: alpha — single metadata file, two queries.
    alpha = real_test / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    _write_metadata(
        alpha / "metadata_0.json",
        dataset="alpha",
        metadataid=0,
        queries=[
            {
                "qid": 100,
                "question": "Is x positively correlated with y?",
                "question_type": "general",
                "difficulty": 1,
            },
            {"qid": 101, "question": "Does x cause y?", "question_type": "causal", "difficulty": 2},
        ],
        domain_knowledge="alphas are special.",
        workflow_tags=["correlation", "regression"],
    )

    # Task 2: beta — two metadata files, one query each.
    beta = real_test / "beta"
    beta.mkdir(parents=True)
    (beta / "data.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    _write_metadata(
        beta / "metadata_0.json",
        dataset="beta",
        metadataid=0,
        queries=[{"qid": 200, "question": "Q200?", "question_type": "general", "difficulty": 1}],
    )
    _write_metadata(
        beta / "metadata_3.json",
        dataset="beta",
        metadataid=3,
        queries=[{"qid": 201, "question": "Q201?", "question_type": "general", "difficulty": 2}],
    )

    return bench_root


# --------------------------------------------------------------------------- #
# discover_tasks                                                              #
# --------------------------------------------------------------------------- #


class TestDiscoverTasks:
    def test_expands_multi_metadata_and_multi_query(self, fake_bench_root: Path):
        tasks = list(mod.discover_tasks(fake_bench_root, "real", "test"))
        triples = sorted((t.dataset, t.metadataid, t.qid) for t in tasks)
        assert triples == [
            ("alpha", 0, 100),
            ("alpha", 0, 101),
            ("beta", 0, 200),
            ("beta", 3, 201),
        ]
        # uid format check
        uids = sorted(t.uid for t in tasks)
        assert uids == [
            "alpha__m0__q100",
            "alpha__m0__q101",
            "beta__m0__q200",
            "beta__m3__q201",
        ]

    def test_resolves_csv_paths(self, fake_bench_root: Path):
        tasks = list(mod.discover_tasks(fake_bench_root, "real", "test"))
        for t in tasks:
            for p in t.csv_paths:
                assert p.is_file()

    def test_missing_split_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(mod.discover_tasks(tmp_path, "real", "test"))

    def test_nested_queries_list_of_lists(self, tmp_path: Path):
        """Real-test metadata stores queries as a list of lists. Confirm
        discover_tasks flattens both shapes."""
        bench = tmp_path / "bench"
        d = bench / "real" / "test" / "nested"
        d.mkdir(parents=True)
        (d / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        meta = {
            "id": 0,
            "domain": "nested",
            "datasets": [
                {
                    "name": "data.csv",
                    "description": "x",
                    "columns": [{"name": "x", "description": "x"}],
                }
            ],
            "queries": [
                [
                    {"qid": 5, "question": "Q5?", "question_type": "context"},
                    {"qid": 6, "question": "Q6?", "question_type": "context"},
                ],
                [{"qid": 7, "question": "Q7?", "question_type": "general"}],
            ],
        }
        (d / "metadata_0.json").write_text(json.dumps(meta), encoding="utf-8")

        tasks = list(mod.discover_tasks(bench, "real", "test"))
        qids = sorted(t.qid for t in tasks)
        assert qids == [5, 6, 7]

    def test_missing_csv_raises(self, tmp_path: Path):
        # task with metadata that points at a nonexistent CSV.
        bench = tmp_path / "bench"
        d = bench / "synth" / "test" / "broken"
        d.mkdir(parents=True)
        _write_metadata(
            d / "metadata_0.json",
            dataset="broken",
            metadataid=0,
            queries=[{"qid": 1, "question": "?", "question_type": "general", "difficulty": 1}],
        )
        # don't create data.csv
        with pytest.raises(FileNotFoundError):
            list(mod.discover_tasks(bench, "synth", "test"))


# --------------------------------------------------------------------------- #
# Result.json validation                                                      #
# --------------------------------------------------------------------------- #


class TestIsValidResultJson:
    def test_good_result(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"hypothesis": "x ↑ ⇒ y ↑", "workflow": "1. ..."}))
        assert mod.is_valid_result_json(p) is True

    def test_workflow_can_be_empty(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"hypothesis": "h", "workflow": ""}))
        assert mod.is_valid_result_json(p) is True

    def test_missing_file_rejected(self, tmp_path: Path):
        assert mod.is_valid_result_json(tmp_path / "nope.json") is False

    def test_empty_file_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text("")
        assert mod.is_valid_result_json(p) is False

    def test_malformed_json_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text("{not json")
        assert mod.is_valid_result_json(p) is False

    def test_empty_hypothesis_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"hypothesis": "", "workflow": "x"}))
        assert mod.is_valid_result_json(p) is False

    def test_whitespace_hypothesis_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"hypothesis": "   ", "workflow": "x"}))
        assert mod.is_valid_result_json(p) is False

    def test_missing_hypothesis_key_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"workflow": "x"}))
        assert mod.is_valid_result_json(p) is False

    def test_non_string_hypothesis_rejected(self, tmp_path: Path):
        p = tmp_path / "result.json"
        p.write_text(json.dumps({"hypothesis": 42}))
        assert mod.is_valid_result_json(p) is False


class TestScanCompletedUids:
    def test_only_valid_uids_are_returned(self, tmp_path: Path):
        out = tmp_path / "out"
        good = out / "good_uid"
        good.mkdir(parents=True)
        (good / mod.RESULT_FILENAME).write_text(json.dumps({"hypothesis": "h", "workflow": "w"}))
        bad = out / "bad_uid"
        bad.mkdir()
        (bad / mod.RESULT_FILENAME).write_text("not json")
        empty = out / "empty_uid"
        empty.mkdir()
        (empty / mod.RESULT_FILENAME).write_text("")
        no_result = out / "no_result_uid"
        no_result.mkdir()

        completed = mod.scan_completed_uids(out)
        assert completed == {"good_uid"}

    def test_missing_root_returns_empty_set(self, tmp_path: Path):
        assert mod.scan_completed_uids(tmp_path / "nope") == set()


# --------------------------------------------------------------------------- #
# Workspace staging                                                           #
# --------------------------------------------------------------------------- #


class TestStageInputs:
    def test_creates_symlinks(self, tmp_path: Path):
        a = tmp_path / "a.csv"
        a.write_text("x\n1\n")
        b = tmp_path / "b.csv"
        b.write_text("y\n2\n")
        ws = tmp_path / "ws"

        names = mod._stage_inputs(ws, [a, b])
        assert names == ["a.csv", "b.csv"]
        for n, src in zip(names, [a, b]):
            link = ws / mod.INPUTS_SUBDIR / n
            assert link.is_symlink()
            assert link.resolve() == src.resolve()

    def test_overwrites_stale_links(self, tmp_path: Path):
        old = tmp_path / "old.csv"
        old.write_text("a")
        new = tmp_path / "new.csv"
        new.write_text("b")
        ws = tmp_path / "ws"
        # Pre-create a stale link with the same target name as the new file.
        (ws / mod.INPUTS_SUBDIR).mkdir(parents=True)
        (ws / mod.INPUTS_SUBDIR / "new.csv").symlink_to(old)

        mod._stage_inputs(ws, [new])
        link = ws / mod.INPUTS_SUBDIR / "new.csv"
        assert link.resolve() == new.resolve()


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


class TestBuildUserQuery:
    def _make_task(self, **overrides) -> mod.Task:
        defaults = dict(
            uid="alpha__m0__q100",
            split_type="real",
            split="test",
            dataset="alpha",
            metadataid=0,
            qid=100,
            metadata_path=Path("/tmp/metadata_0.json"),
            metadata={
                "domain": "alpha",
                "datasets": [
                    {
                        "name": "data.csv",
                        "description": "alpha data",
                        "columns": [
                            {"name": "x", "description": "x col"},
                            {"name": "y", "description": "y col"},
                        ],
                    }
                ],
                "domain_knowledge": "alphas are special.",
                "workflow_tags": ["correlation"],
            },
            query_text="Is x positively correlated with y?",
            csv_paths=[Path("/tmp/data.csv")],
        )
        defaults.update(overrides)
        return mod.Task(**defaults)

    def test_required_sections_present(self):
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"])
        # Title + deliverable contract.
        assert "DiscoveryBench: Hypothesis Discovery Task" in prompt
        assert mod.RESULT_FILENAME in prompt
        assert '"hypothesis"' in prompt and '"workflow"' in prompt
        # Data + columns.
        assert "./inputs/data.csv" in prompt
        assert "`x`" in prompt and "`y`" in prompt
        # The actual question.
        assert t.query_text in prompt
        # Output protocol must forbid pip install.
        assert "Do NOT install" in prompt

    def test_no_plot_directive_present(self):
        """The bench is text-only — plot generation wastes tokens and is
        never scored. The output protocol must explicitly forbid it."""
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"])
        assert "Do NOT generate plots" in prompt
        # Must NOT silently encourage plotting via 'if you plot...' phrasing.
        assert "If you plot" not in prompt
        assert "matplotlib.use('Agg')" not in prompt

    def test_evidence_based_section_present(self):
        """workflow must demand concrete numbers + critic must be told what
        to look for. Both have bitten us in real eval runs — pin them."""
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"])
        assert "Evidence-based requirement" in prompt
        # Workflow rigor cues.
        assert "exact column names" in prompt
        assert "Pearson r" in prompt or "concrete statistic" in prompt
        # Critic / approval review checklist explicitly in prompt.
        assert "critic" in prompt.lower() or "approval" in prompt.lower()
        assert "not backed by a computed value" in prompt
        # Self-check before finalize.
        assert "Self-check" in prompt or "self-check" in prompt

    def test_domain_knowledge_off_by_default(self):
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"])
        assert "alphas are special" not in prompt
        assert "Suggested analytical techniques" not in prompt

    def test_domain_knowledge_when_flag_on(self):
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"], use_domain_knowledge=True)
        assert "alphas are special" in prompt

    def test_workflow_tags_when_flag_on(self):
        t = self._make_task()
        prompt = mod._build_user_query(t, ["data.csv"], use_workflow_tags=True)
        assert "Suggested analytical techniques" in prompt
        assert "correlation" in prompt

    def test_synth_ignores_real_only_flags(self):
        t = self._make_task(split_type="synth")
        prompt = mod._build_user_query(
            t, ["data.csv"], use_domain_knowledge=True, use_workflow_tags=True
        )
        assert "alphas are special" not in prompt
        assert "Suggested analytical techniques" not in prompt

    def test_real_columns_nested_under_raw(self):
        # Some real metadata wraps columns in {"raw": [...]}.
        t = self._make_task(
            metadata={
                "domain": "alpha",
                "datasets": [
                    {
                        "name": "data.csv",
                        "description": "alpha data",
                        "columns": {"raw": [{"name": "z", "description": "z col"}]},
                    }
                ],
            }
        )
        prompt = mod._build_user_query(t, ["data.csv"])
        assert "`z`" in prompt


class TestCopyMetadata:
    """The on-disk metadata copy must match what the prompt promises —
    otherwise an agent that reads ``./metadata.json`` directly would see
    fields the prompt withheld."""

    def _meta(self) -> dict:
        return {
            "id": 0,
            "domain": "alpha",
            "datasets": [
                {
                    "name": "data.csv",
                    "description": "x",
                    "columns": [{"name": "x", "description": "x"}],
                }
            ],
            "domain_knowledge": "secret hint",
            "workflow_tags": ["correlation", "regression"],
            "hypotheses": {"main": [{"text": "GOLD HYPO"}], "intermediate": []},
            "intermediate": [{"hint": "leak"}],
        }

    def _written(self, tmp_path: Path, **flags) -> dict:
        mod._copy_metadata(tmp_path, self._meta(), **flags)
        return json.loads((tmp_path / mod.METADATA_FILENAME).read_text())

    def test_default_strips_all_sensitive_fields(self, tmp_path: Path):
        out = self._written(tmp_path)
        assert "domain_knowledge" not in out
        assert "workflow_tags" not in out
        # hypotheses + intermediate ALWAYS stripped (defense-in-depth).
        assert "hypotheses" not in out
        assert "intermediate" not in out
        # Public fields preserved.
        assert out["domain"] == "alpha"
        assert out["datasets"][0]["name"] == "data.csv"

    def test_domain_knowledge_kept_when_flag_on(self, tmp_path: Path):
        out = self._written(tmp_path, use_domain_knowledge=True)
        assert out["domain_knowledge"] == "secret hint"
        # workflow_tags still stripped.
        assert "workflow_tags" not in out

    def test_workflow_tags_kept_when_flag_on(self, tmp_path: Path):
        out = self._written(tmp_path, use_workflow_tags=True)
        assert out["workflow_tags"] == ["correlation", "regression"]
        assert "domain_knowledge" not in out

    def test_hypotheses_always_stripped_even_with_flags(self, tmp_path: Path):
        out = self._written(
            tmp_path,
            use_domain_knowledge=True,
            use_workflow_tags=True,
        )
        # hypotheses field hard-stripped regardless — guards against a
        # future upstream change populating it on test split.
        assert "hypotheses" not in out
        assert "intermediate" not in out


class TestBuildDataSummary:
    def test_lists_each_csv(self):
        t = TestBuildUserQuery._make_task(
            TestBuildUserQuery(), csv_paths=[Path("a.csv"), Path("b.csv")]
        )
        summary = mod._build_data_summary(t, ["a.csv", "b.csv"])
        assert "./inputs/a.csv" in summary and "./inputs/b.csv" in summary
        assert "preinstalled" in summary

    def test_data_phase_scope_is_narrow(self):
        """Data phase must do light EDA only — no inferential analysis,
        no hypothesis formation, no result.json write. Heavy lifting is
        deferred to the experiment phase."""
        t = TestBuildUserQuery._make_task(TestBuildUserQuery())
        summary = mod._build_data_summary(t, ["data.csv"])
        # Light EDA scope is named explicitly.
        assert "light EDA" in summary or "Light EDA" in summary
        # Forbidden in this phase.
        assert "Do NOT" in summary
        assert "result.json" in summary  # mentioned as withheld
        # The deferral is explicit so the data agent doesn't try to be
        # heroic and steal the experiment phase's work.
        assert "experiment" in summary.lower()


# --------------------------------------------------------------------------- #
# run_one_task wiring                                                          #
# --------------------------------------------------------------------------- #


class TestRunOneTask:
    def _task_for(self, fake_bench_root: Path, dataset: str, metadataid: int, qid: int) -> mod.Task:
        for t in mod.discover_tasks(fake_bench_root, "real", "test"):
            if (t.dataset, t.metadataid, t.qid) == (dataset, metadataid, qid):
                return t
        raise AssertionError(f"task {dataset}/m{metadataid}/q{qid} not found")

    def test_success_path(self, tmp_path: Path, fake_bench_root: Path):
        captured: dict = {}

        def _fake_run(**kwargs):
            captured.update(kwargs)
            ws = Path(kwargs["workspace_path"])
            (ws / mod.RESULT_FILENAME).write_text(
                json.dumps(
                    {"hypothesis": "x and y are correlated.", "workflow": "1. corr.\n2. report."}
                ),
                encoding="utf-8",
            )

            class _Stub:
                final_status = "success"

            return _Stub()

        task = self._task_for(fake_bench_root, "alpha", 0, 100)
        with patch.object(mod, "run_full_workflow", _fake_run):
            record = mod.run_one_task(
                task=task,
                output_root=tmp_path / "out",
                max_revisions=1,
                data_recursion_limit=10,
                exp_recursion_limit=20,
                use_domain_knowledge=False,
                use_workflow_tags=False,
            )

        assert record["ok"] is True
        assert record["uid"] == "alpha__m0__q100"
        assert record["dataset"] == "alpha"
        assert record["metadataid"] == 0
        assert record["qid"] == 100
        assert record["hypothesis"].startswith("x and y")
        assert record["workflow"].startswith("1. corr")
        assert Path(record["result_path"]).exists()

        # Plumbing checks: kwargs reached the workflow layer.
        assert captured["max_revisions"] == 1
        assert captured["data_agent_recursion_limit"] == 10
        assert captured["experiment_agent_recursion_limit"] == 20
        assert captured["user_approval_enabled"] is False
        # data_path points at the workspace's inputs/ dir (not the upstream CSV).
        assert Path(captured["data_path"]).name == mod.INPUTS_SUBDIR

        # task.md was persisted.
        task_md = (tmp_path / "out" / record["uid"] / mod.TASK_PROMPT_FILENAME).read_text()
        assert "DiscoveryBench" in task_md
        assert task.query_text in task_md
        # metadata.json was copied.
        assert (tmp_path / "out" / record["uid"] / mod.METADATA_FILENAME).is_file()

    def test_missing_result_is_failure(self, tmp_path: Path, fake_bench_root: Path):
        def _fake_run(**kwargs):
            class _Stub:
                final_status = "success"

            return _Stub()

        task = self._task_for(fake_bench_root, "alpha", 0, 100)
        with patch.object(mod, "run_full_workflow", _fake_run):
            record = mod.run_one_task(
                task=task,
                output_root=tmp_path / "out",
                max_revisions=1,
                data_recursion_limit=10,
                exp_recursion_limit=20,
                use_domain_knowledge=False,
                use_workflow_tags=False,
            )
        assert record["ok"] is False
        assert "was not created" in record["error"]

    def test_empty_hypothesis_is_failure(self, tmp_path: Path, fake_bench_root: Path):
        def _fake_run(**kwargs):
            ws = Path(kwargs["workspace_path"])
            (ws / mod.RESULT_FILENAME).write_text(
                json.dumps({"hypothesis": "", "workflow": "did nothing"}),
                encoding="utf-8",
            )

            class _Stub:
                final_status = "success"

            return _Stub()

        task = self._task_for(fake_bench_root, "alpha", 0, 100)
        with patch.object(mod, "run_full_workflow", _fake_run):
            record = mod.run_one_task(
                task=task,
                output_root=tmp_path / "out",
                max_revisions=1,
                data_recursion_limit=10,
                exp_recursion_limit=20,
                use_domain_knowledge=False,
                use_workflow_tags=False,
            )
        assert record["ok"] is False
        assert "malformed" in record["error"] or "empty hypothesis" in record["error"]


# --------------------------------------------------------------------------- #
# Roles yaml                                                                  #
# --------------------------------------------------------------------------- #


class TestRegisterModelsFromYaml:
    def test_roles_yaml_exists(self):
        assert mod.ROLES_YAML_PATH.is_file()

    def test_register_models_loads_without_error(self):
        # Just verify it doesn't raise — full role registration is exercised
        # by other tests; we only need to confirm our yaml parses.
        mod._register_models_from_yaml()
