"""A tiny run of analysis/coupling/study.py: tables, compatibility and provenance."""
import importlib.util
import json

import numpy as np

from sdm.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("coupling_study", REPO_ROOT / "analysis" / "coupling" / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_tiny_study_produces_tables_and_provenance(tmp_path):
    results = study.run_study(model_names=["ridge2", "extratrees"], optimizer_names=["pgd", "trust-dirichlet"],
                              truth_names=["extratrees"], seeds=1,
                              optimizer_params={"trust-dirichlet": {"iterations": 3}})
    compatible = results["compatibility"].set_index(["model", "optimizer"]).compatible
    assert not compatible[("extratrees", "pgd")] and compatible[("ridge2", "pgd")]
    synthetic = results["semi_synthetic"]
    assert np.isfinite(synthetic[synthetic.compatible].regret).all()
    assert {"judged_by_ridge2", "judged_by_extratrees", "distance_ratio"} <= set(results["cross_judge"].columns)
    out = study.write(results, tmp_path / "out")
    assert json.loads((out / "provenance.json").read_text())["dataset"]["sha256"].startswith("877b5f8e")
    assert "Semi-synthetic regret" in (out / "summary.md").read_text()
