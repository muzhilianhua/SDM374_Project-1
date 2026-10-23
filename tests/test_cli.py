"""End-to-end CLI: runs are recorded, reproducible, and reusable across commands."""
import json

from sdm.cli import main
from sdm.runs import list_runs
from tests.conftest import BASELINE_CONFIG


def run_cli(tmp_path, *args):
    return main(["--runs-dir", str(tmp_path), *args])


def test_train_then_optimize_from_model_run(tmp_path, capsys):
    assert run_cli(tmp_path, "train", "-c", str(BASELINE_CONFIG)) == 0
    (train_run,) = list_runs(tmp_path)
    meta = train_run.read_json("meta.json")
    assert meta["dataset"]["sha256"].startswith("877b5f8e")
    assert {"commit", "dirty"} <= set(meta["git"])
    assert train_run.read_json("metrics.json")["alpha"] == 100.0

    assert run_cli(tmp_path, "optimize", "-c", str(BASELINE_CONFIG), "--model-run", train_run.id, "-n", "3") == 0
    (optimize_run,) = [run for run in list_runs(tmp_path) if run.read_json("meta.json")["kind"] == "optimize"]
    assert len(optimize_run.read_json("weights.json")) == 3
    assert len(optimize_run.read_json("proposals.json")) == 3

    weights = tmp_path / "w.json"
    weights.write_text(json.dumps(optimize_run.read_json("weights.json")[0]), encoding="utf-8")
    capsys.readouterr()
    assert run_cli(tmp_path, "predict", "--model-run", train_run.id, "--weights", str(weights)) == 0
    predicted = json.loads(capsys.readouterr().out)["predicted_mean"][0]
    assert predicted == optimize_run.read_json("proposals.json")[0]["predicted_mean"]


def test_reproduce_matches_original(tmp_path, capsys):
    assert run_cli(tmp_path, "optimize", "-c", str(BASELINE_CONFIG)) == 0
    (original,) = list_runs(tmp_path)
    assert run_cli(tmp_path, "reproduce", original.id) == 0
    assert "reproduced" in capsys.readouterr().out


def test_reproduce_detects_changed_outputs(tmp_path, capsys):
    assert run_cli(tmp_path, "train", "-c", str(BASELINE_CONFIG)) == 0
    (original,) = list_runs(tmp_path)
    metrics = original.read_json("metrics.json")
    original.write_json("metrics.json", {**metrics, "cv_mae": metrics["cv_mae"] + 0.01})
    assert run_cli(tmp_path, "reproduce", original.id) == 1
    assert "cv_mae" in capsys.readouterr().out


def test_evaluate_reports_shared_protocol_metrics(tmp_path):
    assert run_cli(tmp_path, "evaluate", "-c", str(BASELINE_CONFIG)) == 0
    metrics = list_runs(tmp_path)[0].read_json("metrics.json")
    assert {"mae", "rmse", "r2", "spearman", "top50_recall"} <= set(metrics)


def test_config_errors_exit_cleanly_without_creating_runs(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[model]\nname = "ridge2"\n[model.params]\nalpha = 1\n', encoding="utf-8")
    assert run_cli(tmp_path / "runs", "train", "-c", str(bad)) == 2
    assert "unknown parameter" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
