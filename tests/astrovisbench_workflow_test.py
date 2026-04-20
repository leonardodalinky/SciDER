"""Unit tests for the pure parts of the AstroVisBench workflow.

We don't actually invoke the experiment agent here — that requires an LLM
key and hours of compute. Instead we pin the glue logic that couldn't be
caught by a runtime error: pickle discovery, symlink staging, prompt layout,
PNG validation / scanning, and ``run_one_query`` wiring (via a patched
``run_experiment_workflow``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from bench_workflows import astrovisbench_workflow as mod

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_query() -> dict:
    return {
        "uid": "abc-123",
        "nb_path": "Some_Notebook.ipynb",
        "setup_query": "# Setup\nImports for analysis",
        "setup_gt_code": "import numpy as np\nimport matplotlib.pyplot as plt",
        "processing_query": "# Processing\nCompute the light curves",
        "processing_underspecifications": "Assume normalization is per-star.",
        "processing_gt_code": "train_data = np.arange(100)\ntrain_labels = np.zeros(100)",
        "visualization_query": "# Visualization\nPlot a 4x4 grid of curves.",
        "visualization_underspecifications": "Use red for flares, black otherwise.",
    }


@pytest.fixture
def cache_with_pickles(tmp_path: Path, fake_query) -> Path:
    """Create <cache_root>/<uid>/ with two .pkl files and one ._* garbage file."""
    cache_root = tmp_path / "cache"
    uid_dir = cache_root / fake_query["uid"]
    uid_dir.mkdir(parents=True)
    with (uid_dir / "train_data.pkl").open("wb") as f:
        pickle.dump([1, 2, 3], f)
    with (uid_dir / "train_labels.pkl").open("wb") as f:
        pickle.dump([0, 1, 0], f)
    # macOS metadata — must be filtered out.
    (uid_dir / "._train_data.pkl").write_bytes(b"junk")
    # The executed-notebook file — also must be filtered out.
    (uid_dir / f"{fake_query['uid']}.ipynb").write_text("{}")
    return cache_root


# --------------------------------------------------------------------------- #
# Pickle resolution & staging                                                 #
# --------------------------------------------------------------------------- #


class TestResolveCachePickles:
    def test_returns_only_real_pkls(self, cache_with_pickles: Path, fake_query):
        pkls = mod._resolve_cache_pickles(cache_with_pickles, fake_query["uid"])
        names = sorted(p.name for p in pkls)
        assert names == ["train_data.pkl", "train_labels.pkl"]

    def test_missing_uid_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            mod._resolve_cache_pickles(tmp_path, "does-not-exist")

    def test_empty_uid_dir_raises(self, tmp_path: Path):
        (tmp_path / "uid").mkdir()
        with pytest.raises(FileNotFoundError):
            mod._resolve_cache_pickles(tmp_path, "uid")


class TestStageInputs:
    def test_creates_symlinks_and_returns_varnames(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query
    ):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        pkls = mod._resolve_cache_pickles(cache_with_pickles, fake_query["uid"])
        names = mod._stage_inputs(workspace, pkls)
        assert sorted(names) == ["train_data", "train_labels"]

        inputs_dir = workspace / mod.INPUTS_SUBDIR
        assert inputs_dir.is_dir()
        for pkl in pkls:
            staged = inputs_dir / pkl.name
            assert staged.is_symlink()
            assert staged.resolve() == pkl.resolve()

    def test_overwrites_existing_stale_symlinks(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query
    ):
        """Resuming / re-running must replace dangling or wrong-target links."""
        workspace = tmp_path / "ws"
        inputs_dir = workspace / mod.INPUTS_SUBDIR
        inputs_dir.mkdir(parents=True)
        # Pre-existing link pointing at a bogus path.
        (inputs_dir / "train_data.pkl").symlink_to(tmp_path / "nowhere.pkl")

        pkls = mod._resolve_cache_pickles(cache_with_pickles, fake_query["uid"])
        mod._stage_inputs(workspace, pkls)

        restaged = inputs_dir / "train_data.pkl"
        assert restaged.resolve().exists()
        assert restaged.resolve().parent.name == fake_query["uid"]


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #


class TestBuildUserQuery:
    def test_contains_all_recipe_sections(self, fake_query):
        q = mod._build_user_query(fake_query, var_names=["train_data", "train_labels"])
        # README §"Visualization" recipe inputs are all present.
        assert fake_query["setup_query"].strip().splitlines()[0] in q
        assert "import numpy as np" in q
        assert "Processing" in q  # processing_query + code labelled
        assert "Processing underspecifications" in q
        assert "per-star" in q  # from processing_underspec
        # Task + vis underspec included.
        assert "Plot a 4x4 grid" in q
        assert "Visualization underspecifications" in q
        assert "red for flares" in q
        # Pickle recipe.
        assert "./inputs/train_data.pkl" in q
        assert "./inputs/train_labels.pkl" in q
        # Output contract.
        assert mod.OUTPUT_IMAGE_FILENAME in q
        assert "plt.savefig" in q

    def test_empty_underspecs_are_elided(self, fake_query):
        q = dict(fake_query)
        q["processing_underspecifications"] = ""
        q["visualization_underspecifications"] = ""
        out = mod._build_user_query(q, var_names=["train_data"])
        assert "Processing underspecifications" not in out
        assert "Visualization underspecifications" not in out

    def test_core_dont_do_this_instructions_present(self, fake_query):
        """The prompt must still tell the agent: no package installs, no
        re-running processing. (Network/filesystem access is now allowed
        via ``bench_env_root``.)"""
        q = mod._build_user_query(fake_query, var_names=["x"])
        low = q.lower()
        assert "do not install" in low
        assert "do not re-run" in low

    def test_bench_env_section_absent_by_default(self, fake_query):
        """No --bench-env-root flag → no mention of bench environment."""
        q = mod._build_user_query(fake_query, var_names=["x"])
        assert "bench_env" not in q.lower()
        assert "Additional resource" not in q

    def test_bench_env_section_included_when_root_provided(self, fake_query):
        """With --bench-env-root set, prompt points at the right subdir and
        explains the _comped.tar.gz + .ipynb layout."""
        fake_query["nb_path"] = "Classifying_TESS_flares_with_CNNs.ipynb"
        q = mod._build_user_query(
            fake_query,
            var_names=["x"],
            bench_env_root=Path("/opt/bench_environment"),
        )
        # Section headline.
        assert "Additional resource" in q
        # Per-nb subdir path is rendered correctly.
        assert "/opt/bench_environment/Classifying_TESS_flares_with_CNNs/" in q
        # Tarball name + how to extract.
        assert "Classifying_TESS_flares_with_CNNs_comped.tar.gz" in q
        assert "tarfile" in q and "extractall" in q
        # Still forbids HTTP — local bench_env is the escape hatch, not downloads.
        assert "do not download" in q.lower() or "NOT download" in q


class TestBuildDataSummary:
    def test_lists_all_vars(self):
        s = mod._build_data_summary(["a", "b"])
        assert "`a`" in s and "`b`" in s
        assert "./inputs/a.pkl" in s and "./inputs/b.pkl" in s
        # Agent is told to NOT install anything.
        assert "do not install" in s.lower()


# --------------------------------------------------------------------------- #
# Per-query driver (with experiment_workflow patched out)                     #
# --------------------------------------------------------------------------- #


class TestRunOneQuery:
    def test_success_path_records_image_and_uses_config(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query
    ):
        captured_kwargs: dict = {}

        def _fake_run(**kwargs):
            captured_kwargs.update(kwargs)
            # Simulate the agent producing the image.
            (Path(kwargs["workspace_path"]) / mod.OUTPUT_IMAGE_FILENAME).write_bytes(
                b"\x89PNG\r\n\x1a\nnot-a-real-png-but-nonzero"
            )

            class _Stub:
                final_status = "success"

            return _Stub()

        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)

        with patch.object(mod, "run_experiment_workflow", _fake_run):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=venv,
                max_revisions=1,
                recursion_limit=50,
            )

        assert record["ok"] is True
        assert record["uid"] == fake_query["uid"]
        assert record["image_path"].endswith(mod.OUTPUT_IMAGE_FILENAME)
        assert Path(record["image_path"]).exists()

        # Plumbing checks for Phases 1–3: config reached the workflow layer.
        cfg = captured_kwargs["workspace_init_config"]
        assert cfg.env_manager == "python"
        assert cfg.init_uv is False
        assert Path(cfg.venv_path).resolve() == venv.resolve()
        assert captured_kwargs["max_revisions"] == 1
        assert captured_kwargs["user_approval_enabled"] is False

        # task.md was persisted next to the image.
        task_md = (tmp_path / "out" / fake_query["uid"] / "task.md").read_text()
        assert "Visualization Task" in task_md

    def test_missing_image_is_failure(self, tmp_path: Path, cache_with_pickles: Path, fake_query):
        def _fake_run(**kwargs):
            # Agent "finishes" but doesn't write the image.
            class _Stub:
                final_status = "success"

            return _Stub()

        with patch.object(mod, "run_experiment_workflow", _fake_run):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=None,
                max_revisions=1,
                recursion_limit=50,
            )

        assert record["ok"] is False
        assert "was not created" in record["error"]

    def test_empty_image_is_failure(self, tmp_path: Path, cache_with_pickles: Path, fake_query):
        def _fake_run(**kwargs):
            (Path(kwargs["workspace_path"]) / mod.OUTPUT_IMAGE_FILENAME).write_bytes(b"")

            class _Stub:
                final_status = "success"

            return _Stub()

        with patch.object(mod, "run_experiment_workflow", _fake_run):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=None,
                max_revisions=1,
                recursion_limit=50,
            )

        assert record["ok"] is False
        assert "empty" in record["error"].lower()

    def test_missing_pickle_dir_is_failure_not_crash(self, tmp_path: Path, fake_query):
        # Cache root exists but has no uid subdir — the driver should record
        # an error, not raise.
        record = mod.run_one_query(
            query=fake_query,
            cache_root=tmp_path / "empty-cache",
            output_root=tmp_path / "out",
            astrovis_venv=None,
            max_revisions=1,
            recursion_limit=50,
        )
        assert record["ok"] is False
        assert fake_query["uid"] in record["error"] or "cache" in record["error"].lower()


# --------------------------------------------------------------------------- #
# PNG validation + completed-uid scan (backs --skip-existing)                 #
# --------------------------------------------------------------------------- #


_REAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"  # magic
    b"\x00\x00\x00\rIHDR"  # IHDR chunk header (length + name)
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestIsValidPng:
    def test_valid_png_detected(self, tmp_path: Path):
        p = tmp_path / "ok.png"
        p.write_bytes(_REAL_PNG_BYTES)
        assert mod.is_valid_png(p) is True

    def test_empty_file_rejected(self, tmp_path: Path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        assert mod.is_valid_png(p) is False

    def test_wrong_magic_rejected(self, tmp_path: Path):
        p = tmp_path / "jpg.png"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 100)  # JPEG magic
        assert mod.is_valid_png(p) is False

    def test_missing_file_rejected(self, tmp_path: Path):
        assert mod.is_valid_png(tmp_path / "nope.png") is False


class TestScanCompletedUids:
    def test_returns_uids_with_valid_png(self, tmp_path: Path):
        # uid-A has valid PNG; uid-B has an empty file; uid-C has nothing.
        (tmp_path / "uid-A").mkdir()
        (tmp_path / "uid-A" / mod.OUTPUT_IMAGE_FILENAME).write_bytes(_REAL_PNG_BYTES)

        (tmp_path / "uid-B").mkdir()
        (tmp_path / "uid-B" / mod.OUTPUT_IMAGE_FILENAME).write_bytes(b"")

        (tmp_path / "uid-C").mkdir()

        # A loose file (not a dir) at the root must be ignored.
        (tmp_path / "results.json").write_text("[]")

        assert mod.scan_completed_uids(tmp_path) == {"uid-A"}

    def test_missing_root_returns_empty(self, tmp_path: Path):
        assert mod.scan_completed_uids(tmp_path / "does-not-exist") == set()


# --------------------------------------------------------------------------- #
# HF loader (hits no network — hf_hub_download is patched)                    #
# --------------------------------------------------------------------------- #


class TestLoadQueriesFromHf:
    def test_reads_json_from_hf_hub_download(self, tmp_path: Path):
        fake_json = tmp_path / "astrovisbench_queries.json"
        fake_json.write_text('[{"uid": "x"}, {"uid": "y"}]', encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_json)) as mock_dl:
            out = mod.load_queries_from_hf()

        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args.kwargs
        assert call_kwargs["repo_id"] == mod.HF_REPO_ID
        assert call_kwargs["filename"] == mod.HF_QUERIES_FILENAME
        assert call_kwargs["repo_type"] == "dataset"
        assert [q["uid"] for q in out] == ["x", "y"]


# --------------------------------------------------------------------------- #
# Role-yaml registration points at the right file                             #
# --------------------------------------------------------------------------- #


class TestRegisterModelsFromYaml:
    def test_roles_yaml_exists(self):
        assert (
            mod.ROLES_YAML_PATH.is_file()
        ), f"Bench-specific role yaml missing at {mod.ROLES_YAML_PATH}"

    def test_registers_from_expected_path(self):
        with patch.object(
            mod, "register_defaults_from_yaml", return_value={"experiment": "x"}
        ) as mock_reg:
            mod._register_models_from_yaml()
        mock_reg.assert_called_once_with(mod.ROLES_YAML_PATH)


# --------------------------------------------------------------------------- #
# validate_pickles (subprocess probe)                                         #
# --------------------------------------------------------------------------- #


class TestValidatePickles:
    """The probe runs under ``python_exec``; we use ``sys.executable`` from
    this test process — it doesn't have upstream classes like
    lightkurve.LightCurveCollection, but that's fine for smoke-testing the
    OK / broken / empty-dict path. Real-world broken pickles (class missing)
    would fail via the AttributeError branch of pickle.load, also caught."""

    def _picklable_temp(self, tmp_path: Path, value: object, name: str) -> Path:
        """Drop a real, loadable pickle into tmp_path."""
        import pickle as _pkl

        p = tmp_path / name
        with p.open("wb") as f:
            _pkl.dump(value, f)
        return p

    def test_all_good_returns_empty_dict(self, tmp_path: Path):
        import sys

        good = [
            self._picklable_temp(tmp_path, [1, 2, 3], "a.pkl"),
            self._picklable_temp(tmp_path, {"x": 1}, "b.pkl"),
        ]
        assert mod.validate_pickles(good, Path(sys.executable)) == {}

    def test_truncated_pickle_is_reported(self, tmp_path: Path):
        import sys

        good = self._picklable_temp(tmp_path, list(range(1000)), "g.pkl")
        bad = tmp_path / "bad.pkl"
        # Keep just a prefix of a real pickle → EOFError on load.
        bad.write_bytes(good.read_bytes()[:20])

        out = mod.validate_pickles([good, bad], Path(sys.executable))
        assert "bad.pkl" in out
        assert "g.pkl" not in out  # the good one must not be reported
        assert "EOFError" in out["bad.pkl"] or "Error" in out["bad.pkl"]

    def test_empty_file_is_reported(self, tmp_path: Path):
        import sys

        empty = tmp_path / "empty.pkl"
        empty.write_bytes(b"")
        out = mod.validate_pickles([empty], Path(sys.executable))
        assert "empty.pkl" in out

    def test_probe_crash_reports_all_broken(self, tmp_path: Path):
        # Nonexistent interpreter → subprocess returns non-zero.
        out = mod.validate_pickles(
            [tmp_path / "anything.pkl"],
            Path("/nonexistent/python"),
        )
        assert "anything.pkl" in out

    def test_pointer_pickles_are_not_rejected(self, tmp_path: Path):
        """Since the agent can now resolve filename/path pickles via
        ``bench_env_root``, ``validate_pickles`` must NOT reject them."""
        import sys
        from pathlib import PosixPath as _PP

        name_pkl = self._picklable_temp(tmp_path, "hst_7656_foo.fits", "stis_coadd_filename.pkl")
        dir_pkl = self._picklable_temp(tmp_path, _PP("stis_products"), "stis_products_dir.pkl")

        out = mod.validate_pickles([name_pkl, dir_pkl], Path(sys.executable))
        assert out == {}, "Pointer pickles must load-OK now — bench_env_root handles them."


# --------------------------------------------------------------------------- #
# run_one_query — broken cache short-circuits BEFORE any LLM call             #
# --------------------------------------------------------------------------- #


class TestRunOneQueryValidation:
    def test_broken_pickle_skips_agent(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query: dict
    ):
        """When validate_pickles flags a file as broken, run_experiment_workflow
        must NOT be called — this is the whole point of the pre-flight check."""
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        # Make the venv's python a real interpreter so the `.exists()` check
        # passes inside run_one_query.
        import sys as _sys

        (venv / "bin" / "python").symlink_to(_sys.executable)

        with (
            patch.object(
                mod,
                "validate_pickles",
                return_value={"train_data.pkl": "EOFError: Ran out of input"},
            ),
            patch.object(mod, "run_experiment_workflow") as mock_run,
        ):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=venv,
                max_revisions=1,
                recursion_limit=50,
            )

        assert record["ok"] is False
        assert "Broken pickles" in record["error"]
        assert "train_data.pkl" in record["error"]
        mock_run.assert_not_called()  # no LLM burn on broken input

    def test_good_pickles_still_run_the_agent(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query: dict
    ):
        """Positive control — clean cache must NOT trip the short-circuit."""
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        import sys as _sys

        (venv / "bin" / "python").symlink_to(_sys.executable)

        def _fake_run(**kwargs):
            (Path(kwargs["workspace_path"]) / mod.OUTPUT_IMAGE_FILENAME).write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"0" * 50
            )

            class _S:
                final_status = "success"

            return _S()

        with (
            patch.object(mod, "validate_pickles", return_value={}),
            patch.object(mod, "run_experiment_workflow", side_effect=_fake_run) as mock_run,
        ):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=venv,
                max_revisions=1,
                recursion_limit=50,
            )

        assert record["ok"] is True
        mock_run.assert_called_once()

    def test_validation_skipped_when_no_astrovis_venv(
        self, tmp_path: Path, cache_with_pickles: Path, fake_query: dict
    ):
        """If no astrovis venv is configured, we can't reliably probe — the
        check is skipped and the agent gets a chance (old behaviour)."""

        def _fake_run(**kwargs):
            (Path(kwargs["workspace_path"]) / mod.OUTPUT_IMAGE_FILENAME).write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"0" * 50
            )

            class _S:
                final_status = "success"

            return _S()

        with (
            patch.object(mod, "validate_pickles") as mock_val,
            patch.object(mod, "run_experiment_workflow", side_effect=_fake_run),
        ):
            record = mod.run_one_query(
                query=fake_query,
                cache_root=cache_with_pickles,
                output_root=tmp_path / "out",
                astrovis_venv=None,  # ← key
                max_revisions=1,
                recursion_limit=50,
            )

        mock_val.assert_not_called()
        assert record["ok"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
