"""Print the online config path for 03_run_online.bat, pointing it at a solver.exe outside the repository.

    python with_solver.py <config name> [<solver.exe path>]

With no path (or an empty one) the config is used as is (repo/solver/solver.exe). With a path, writes
experiments/<config>-here.toml, a copy whose [evaluator.params] gains `exe = '<path>'`, and prints it.
The sha256 check against solver/CHECKSUMS.sha256 still applies when the run starts.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: with_solver.py <config name> [<solver.exe path>]")
        return 2
    source = REPO / "experiments" / f"{argv[0]}.toml"
    if not source.is_file():
        print(f"config not found: {source}")
        return 1
    exe = argv[1].strip().strip('"') if len(argv) > 1 else ""
    if not exe:
        default = REPO / "solver" / "solver.exe"
        if not default.is_file():
            print(f"{default} not found: copy solver.exe there or set SOLVER_EXE in settings.bat")
            return 1
        print(source)
        return 0
    path = Path(exe)
    if not path.is_file():
        print(f"SOLVER_EXE does not exist: {path}")
        return 1
    lines = source.read_text(encoding="utf-8").splitlines()
    if "[evaluator.params]" not in [line.strip() for line in lines]:
        print(f"{source.name} has no [evaluator.params] section")
        return 1
    out = []
    for line in lines:
        out.append(line)
        if line.strip() == "[evaluator.params]":
            out.append(f"exe = '{path.resolve()}'   # set by 03_run_online.bat (settings.bat SOLVER_EXE)")
    target = source.with_name(f"{argv[0]}-here.toml")
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
