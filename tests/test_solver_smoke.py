"""Run analysis/solver_smoke/smoke.py against a fake evaluator that follows solver/README.md."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("solver_smoke", ROOT / "analysis/solver_smoke/smoke.py")

FAKE = '''import json, math, pathlib, sys
def fail(msg):
    print(msg, file=sys.stderr); sys.exit(2)
try:
    data = json.loads(sys.stdin.read())
except ValueError:
    fail("bad json")
w = data["weights"] if isinstance(data, dict) else data
single = (len(w) == 21 and not isinstance(w[0], list)) or (
    len(w) == 3 and isinstance(w[0], list) and not isinstance(w[0][0], list) and len(w[0]) == 7)
sets = [w] if single else w
if not 1 <= len(sets) <= 50:
    fail("1-50 sets")
out = []
for s in sets:
    v = [x for r in s for x in r] if isinstance(s[0], list) else list(s)
    if len(v) != 21 or any(not math.isfinite(x) or x < 0 for x in v):
        fail("bad weights")
    b = [v[i * 7:(i + 1) * 7] for i in range(3)]
    if any(sum(x) <= 0 for x in b):
        fail("positive")
    n = [x / sum(bb) for bb in b for x in bb]
    out.append(114.4317428462673 if all(abs(x - 1 / 7) < 1e-12 for x in n) else 110 + 10 * sum(x * x for x in n))
c = pathlib.Path("usage.count")
k = int(c.read_text()) if c.exists() else 0
if k >= 20:
    fail("quota")
c.write_text(str(k + 1))
print(json.dumps({"R": out}))
'''


@pytest.mark.skipif(sys.platform == "win32", reason="the fake evaluator is a POSIX script")
def test_smoke_against_fake_solver(tmp_path, monkeypatch):
    # A shell wrapper, because a shebang cannot hold an interpreter path with spaces.
    (tmp_path / "fake_solver.py").write_text(FAKE)
    exe = tmp_path / "solver.exe"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/fake_solver.py"\n')
    exe.chmod(0o755)
    report = tmp_path / "report.json"
    smoke = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(smoke)
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--exe", str(exe), "--quick", "--quota", "--out", str(report)])

    assert smoke.main() == 1
    data = json.loads(report.read_text())
    # A fake cannot match the real exe's checksum or the measured R values; everything else must pass.
    assert data["failed"] == ["exe_sha256", "known_points"]
    checks = data["checks"]
    assert checks["readme_example"]["passed"] and checks["timing"]["passed"]
    assert all(checks[name]["passed"] for name in checks if name.startswith("rejects_"))
    assert checks["renormalises_per_building"]["equal_to_normalised"] is True
    assert checks["quota_after_20_calls"]["calls"][-1]["exit_code"] == 2
    assert not (tmp_path / "usage.count").exists()
