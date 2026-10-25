"""Background `sdm` commands for the workbench. No Streamlit here, so it can be tested directly.

Each job lives in <runs_root>/.jobs/<job id>/:
  job.json   command, created time and wrapper pid
  log.txt    combined stdout and stderr
  child.pid  pid of the sdm process (used to cancel it)
  exit_code  written when the command finishes
  cancelled  marker written by cancel()
list_runs ignores this folder because it has no meta.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from sdm.paths import REPO_ROOT

RUNNER = Path(__file__).with_name("run_job.py")
SDM = [sys.executable, "-m", "sdm"]
RUN_LINE = re.compile(r"^(?:run|rerun):\s+(.+)$", re.MULTILINE)

RUNNING, SUCCEEDED, FAILED, CANCELLED = "running", "succeeded", "failed", "cancelled"


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        # Reap our own finished wrappers so they do not linger as zombies that look alive.
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class Job:
    path: Path

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def spec(self) -> dict:
        return json.loads((self.path / "job.json").read_text(encoding="utf-8"))

    @property
    def exit_code(self) -> int | None:
        file = self.path / "exit_code"
        return int(file.read_text(encoding="utf-8")) if file.exists() else None

    @property
    def status(self) -> str:
        code = self.exit_code
        if (self.path / "cancelled").exists() and code != 0:
            return CANCELLED
        if code is not None:
            return SUCCEEDED if code == 0 else FAILED
        pid = self.spec.get("pid")
        # No pid yet means the job was created a moment ago; a dead wrapper without an exit code crashed.
        return RUNNING if pid is None or _alive(pid) else FAILED

    def log(self) -> str:
        file = self.path / "log.txt"
        return file.read_text(encoding="utf-8", errors="replace") if file.exists() else ""

    def tail(self, lines: int = 30) -> str:
        return "\n".join(self.log().splitlines()[-lines:])

    @property
    def run_path(self) -> Path | None:
        """The run folder the command created, parsed from its `run:` / `rerun:` line."""
        matches = RUN_LINE.findall(self.log())
        return Path(matches[-1].strip()) if matches else None


def start(args: list[str], runs_root: Path, *, display: str | None = None,
          program: list[str] | None = None) -> Job:
    """Launch `sdm --runs-dir <runs_root> <args>` in the background and return immediately.

    `program` replaces the `python -m sdm` prefix; tests use it to run a controllable command.
    """
    runs_root = Path(runs_root)
    jobs_dir = runs_root / ".jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{args[0]}"
    for suffix in range(1, 1000):
        path = jobs_dir / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            path.mkdir()
            break
        except FileExistsError:
            continue
    argv = ["--runs-dir", str(runs_root), *args]
    spec = {"args": list(args), "argv": argv, "program": program or SDM, "cwd": str(REPO_ROOT),
            "display": display or " ".join(["sdm", *args]), "created": datetime.now().isoformat(timespec="seconds"),
            "pid": None}
    (path / "job.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
    wrapper = subprocess.Popen([sys.executable, str(RUNNER), str(path)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=REPO_ROOT, **detach)
    (path / "job.json").write_text(json.dumps({**spec, "pid": wrapper.pid}, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return Job(path)


def list_jobs(runs_root: Path) -> list[Job]:
    """Jobs under runs_root, newest first."""
    jobs_dir = Path(runs_root) / ".jobs"
    if not jobs_dir.exists():
        return []
    paths = [path for path in jobs_dir.iterdir() if (path / "job.json").exists()]
    return [Job(path) for path in sorted(paths, key=lambda path: path.name, reverse=True)]


def cancel(job: Job) -> None:
    """Stop a running job; on Windows os.kill with SIGTERM calls TerminateProcess."""
    if job.status != RUNNING:
        return
    (job.path / "cancelled").write_text("", encoding="utf-8")
    child = job.path / "child.pid"
    pid = int(child.read_text(encoding="utf-8")) if child.exists() else job.spec.get("pid")
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def run_now(args: list[str], runs_root: Path, *, input_text: str | None = None,
            timeout: float = 120) -> subprocess.CompletedProcess:
    """Run a quick `sdm` command synchronously (e.g. predict) and capture its output."""
    return subprocess.run([*SDM, "--runs-dir", str(runs_root), *args], input=input_text, capture_output=True,
                          text=True, cwd=REPO_ROOT, timeout=timeout)
