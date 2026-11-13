import hashlib
import json
from pathlib import Path
import shutil

import pytest

from analysis.ablation.build_table import build_table


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "analysis/ablation/evidence"


def test_archived_baselines_rebuild_with_exact_source_hashes():
    manifest = json.loads((EVIDENCE / "CHECKSUMS.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    table = build_table(EVIDENCE)
    assert len(table) == 2
    assert set(table.protocol) == {"frozen-v32", "kfold"}
    assert table.loc[table.protocol == "frozen-v32", "v32_score"].iloc[0] == pytest.approx(66.4507565104)
    assert table.loc[table.protocol == "kfold", "mae"].iloc[0] == pytest.approx(6.8275199704)
    assert all(len(commit) == 40 for commit in table.commit)


@pytest.mark.parametrize("problem", ["dirty", "missing_commit", "missing_config", "changed_config"])
def test_default_excludes_unverified_runs_but_opt_in_preserves_them(tmp_path, problem):
    source = next(EVIDENCE.glob("*-evaluate-baseline-v32"))
    target = tmp_path / "records" / source.name
    shutil.copytree(source, target)
    config = tmp_path / "project/experiments/baseline-v32.toml"
    config.parent.mkdir(parents=True)
    shutil.copyfile(source / "config.toml", config)
    meta_path = target / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if problem == "dirty":
        meta["git"]["dirty"] = True
    elif problem == "missing_commit":
        meta["git"]["commit"] = None
    elif problem == "missing_config":
        config.unlink()
    else:
        config.write_text('[model]\nname = "different"\n', encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert build_table(tmp_path / "records", project_root=tmp_path / "project").empty
    assert len(build_table(tmp_path / "records", project_root=tmp_path / "project",
                           include_unverified=True)) == 1
