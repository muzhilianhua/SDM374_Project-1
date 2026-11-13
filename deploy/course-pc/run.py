"""SDM374 Project 1 online run, in the shape of the course's solver/run.py (T10, temporary version).

The course template calls solver.exe 20 times and leaves "retrain the surrogate, update weight.json" as
todo. This script fills that loop with our online program (`sdm online`, T08): each call proposes up to 50
new weight sets from the surrogate, screens them, sends them to solver.exe, logs the R values, refits on
all observations and repeats, at most 20 calls. It resumes after an interruption without resending.

Standard library only, so any Python (3.8+, including the course computer's 3.14) can start it; the
online program itself runs in the kit's bundled Python 3.11 with the bundled wheels.

    python run.py                 real online run (asks for YES; --yes skips the question)
    python run.py --yes           the same, unattended (graded-run form)
    python run.py probe           record this computer's environment (no solver call)
    python run.py setup           install the bundled wheels into the bundled Python (offline)
    python run.py selftest        tests, 3 calls on a semi-synthetic truth, first-batch dry runs, timing
    python run.py check           probe + setup + selftest in one go (no solver call)
    python run.py resume [RUN]    continue the latest (or given) online run
    python run.py collect         zip results/ to bring back

Options for the real run: --solver PATH (solver.exe; default: next to run.py, else repo/solver/),
--config NAME (repo/experiments/NAME.toml), --max-calls N (stop after N calls in this session).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---- Settings (command-line options override them) -------------------------------------------------
SOLVER_EXE = ""                 # full path to solver.exe; empty: next to run.py, else repo/solver/
ONLINE_CONFIG = "online-histgb"  # repo/experiments/<name>.toml: histgb surrogate + pure-exploit pool
MAX_CALLS = None                # None: up to 20 calls; an int stops this session after that many
# -----------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
if (HERE / "repo" / "src" / "sdm").is_dir():      # unpacked kit: run.py sits next to repo/ and python/
    KIT, REPO = HERE, HERE / "repo"
else:                                             # inside the repository: deploy/course-pc/run.py
    REPO = HERE.parents[1]
    KIT = REPO
RESULTS = KIT / "results"
BUNDLED = KIT / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
TOOLS = REPO / "deploy" / "course-pc"
SETUP_MARK = RESULTS / ".setup-ok"


def say(text: str) -> None:
    print(text, flush=True)


def runtime_python() -> str:
    """The Python 3.11 that runs `sdm`: the bundled one, else this one when it is 3.11."""
    if BUNDLED.exists():
        return str(BUNDLED)
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    raise SystemExit(f"No Python 3.11: {BUNDLED} is missing and this is Python {sys.version.split()[0]}. "
                     "Use the full kit (it bundles Python 3.11).")


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return env


def execute(command: list[str], log: Path | None = None, *, quiet: bool = False) -> int:
    """Run in repo/, echo the output (unless quiet) and copy it to `log`."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8") if log else None
    try:
        if handle:
            handle.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=REPO, env=child_env(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            if not quiet:
                print(line, end="", flush=True)
            if handle:
                handle.write(line)
        return process.wait()
    finally:
        if handle:
            handle.close()


def sdm(*args: str) -> list[str]:
    return [runtime_python(), "-m", "sdm", "--runs-dir", str(RESULTS / "runs"), *args]


# ---- steps ------------------------------------------------------------------------------------------

def step_probe(_args) -> int:
    say("[probe] Recording this computer's environment (no solver call)...")
    return execute([sys.executable, str(TOOLS / "probe_env.py"), "--out", str(RESULTS)], RESULTS / "probe.log")


def step_setup(_args) -> int:
    python = runtime_python()
    if python == sys.executable and not BUNDLED.exists():
        say("[setup] Using this Python 3.11 as is (no bundled Python to install into).")
    else:
        say(f"[setup] Installing the bundled wheels into {python} (offline)...")
        code = execute([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
                        "--find-links", str(KIT / "wheels"), "-r", str(TOOLS / "requirements-win.txt")],
                       RESULTS / "setup.log", quiet=True)
        if code:
            say(f"[setup] pip install FAILED (exit {code}); see {RESULTS / 'setup.log'}")
            return code
    code = execute(sdm("check-data", "--json"), RESULTS / "setup.log")
    if code:
        say("[setup] check-data FAILED")
        return code
    execute(sdm("list"), RESULTS / "setup.log", quiet=True)
    SETUP_MARK.write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    say("[setup] OK")
    return 0


def ensure_setup() -> int:
    return 0 if SETUP_MARK.exists() else step_setup(None)


def step_selftest(_args) -> int:
    if ensure_setup():
        return 1
    python = runtime_python()
    checks = [
        ("1/4 test suite", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"], "selftest_pytest.log"),
        ("2/4 3 online calls on a semi-synthetic truth (no solver)",
         sdm("online", "-c", "experiments/online-histgb-synthetic.toml", "--max-calls", "3"), "selftest_synthetic.log"),
        ("3/4 first-batch dry run of online-histgb (calls nothing)",
         sdm("online", "-c", f"experiments/{ONLINE_CONFIG}.toml", "--dry-run"), "selftest_dryrun.log"),
        ("4/4 surrogate fit timing and prediction fingerprints",
         [python, str(TOOLS / "bench_models.py"), "--out", str(RESULTS), "--configs", "baseline-v32",
          "histgb-v32", "xgboost-v32", "extratrees-v32", "local-gp-v32", "online-histgb"], "selftest_bench.log"),
    ]
    failed = 0
    for label, command, log in checks:
        say(f"[selftest] {label}...")
        start = time.perf_counter()
        code = execute(command, RESULTS / log, quiet=True)
        lines = (RESULTS / log).read_text(encoding="utf-8", errors="replace").strip().splitlines()
        say(f"           exit {code} in {time.perf_counter() - start:.0f} s; last line: {lines[-1] if lines else ''}")
        failed += code != 0
    say(f"[selftest] Done, {failed} step(s) failed. Logs in {RESULTS}")
    return 1 if failed else 0


def find_solver(explicit: str) -> Path | None:
    for candidate in ([Path(explicit)] if explicit else []) + [HERE / "solver.exe", REPO / "solver" / "solver.exe"]:
        if candidate.is_file():
            return candidate.resolve()
    return None


def step_online(args) -> int:
    if ensure_setup():
        return 1
    solver = find_solver(args.solver or SOLVER_EXE)
    if solver is None:
        say("[online] solver.exe not found. Put it next to run.py, in repo/solver/, or pass --solver PATH.")
        return 1
    made = subprocess.run([sys.executable, str(TOOLS / "with_solver.py"), args.config, str(solver)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if made.returncode:
        say(made.stdout + made.stderr)
        return made.returncode
    config = made.stdout.strip().splitlines()[-1]
    limit = ["--max-calls", str(args.max_calls)] if args.max_calls else []
    say("=" * 66)
    say(" REAL ONLINE RUN: calls solver.exe. Every call uses one of its 20")
    say(" calls (usage.count next to the exe), even when the call fails.")
    say(f" solver : {solver}")
    say(f" config : {args.config}")
    say(f" calls  : {args.max_calls or 'up to 20'} in this session")
    say("=" * 66)
    if not args.yes and input("Type YES to start: ").strip().upper() != "YES":
        say("Cancelled.")
        return 1
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    code = execute(sdm("online", "-c", config, "--name", args.config, *limit), RESULTS / f"online-{stamp}.log")
    say(f"[online] exit {code}. Run folders: {RESULTS / 'runs'}. Interrupted? python run.py resume")
    return code


def calls_solver(run: Path) -> bool:
    """True when the run folder's saved config sends batches to solver.exe (not a replay truth)."""
    try:
        text = (run / "config.toml").read_text(encoding="utf-8")
    except OSError:
        return False
    section = re.search(r"^\[evaluator\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    return bool(section and re.search(r'^\s*kind\s*=\s*["\']solver["\']', section.group(1), re.M))


def step_resume(args) -> int:
    if ensure_setup():
        return 1
    run = args.run
    if not run:
        folders = sorted((p for p in (RESULTS / "runs").glob("*-online-*") if p.is_dir()
                          and (p / "attempts.jsonl").exists() and calls_solver(p)), key=lambda p: p.name)
        if not folders:
            say(f"[resume] No solver run to continue in {RESULTS / 'runs'} (replay and dry runs are skipped)")
            return 1
        run = str(folders[-1])
    if not args.yes and input(f"Resume {Path(run).name} (calls solver.exe)? Type YES: ").strip().upper() != "YES":
        say("Cancelled.")
        return 1
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return execute(sdm("online", "--resume", run), RESULTS / f"resume-{stamp}.log")


def step_collect(_args) -> int:
    name = f"results-{os.environ.get('COMPUTERNAME', 'pc')}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    archive = shutil.make_archive(str(KIT / name), "zip", RESULTS)
    say(f"[collect] Copy this file back: {archive}")
    return 0


def step_check(args) -> int:
    codes = [step_probe(args), step_setup(args), step_selftest(args)]
    say(f"[check] exit codes probe/setup/selftest: {codes}. Next: python run.py collect (or the real run)")
    return max(codes)


def main() -> int:
    parser = argparse.ArgumentParser(description="SDM374 online run (course run.py form)",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("step", nargs="?", default="online",
                        choices=["online", "probe", "setup", "selftest", "check", "resume", "collect"])
    parser.add_argument("run", nargs="?", help="resume: run id or folder (default: the latest)")
    parser.add_argument("--solver", default="", help="path to solver.exe")
    parser.add_argument("--config", default=ONLINE_CONFIG, help="config name in repo/experiments")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--yes", action="store_true", help="do not ask before calling solver.exe")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    say(f"SDM374 online kit  |  code {(REPO / 'VERSION').read_text().split()[0][:7] if (REPO / 'VERSION').exists() else 'dev'}"
        f"  |  launcher Python {sys.version.split()[0]}  |  results: {RESULTS}")
    steps = {"online": step_online, "probe": step_probe, "setup": step_setup, "selftest": step_selftest,
             "check": step_check, "resume": step_resume, "collect": step_collect}
    return steps[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
