"""Run one workbench job: `python run_job.py <job_dir>`.

Reads the command from <job_dir>/job.json, streams its output to <job_dir>/log.txt and writes the exit
code to <job_dir>/exit_code. It is a separate process, not a Streamlit thread, so a job survives page
reruns, navigation and browser reloads.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    job_dir = Path(sys.argv[1])
    spec = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    # Unbuffered so the log fills while the command runs, not only when it exits.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(job_dir / "log.txt", "wb") as log:
        child = subprocess.Popen([*spec["program"], *spec["argv"]], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, cwd=spec["cwd"], env=env)
        (job_dir / "child.pid").write_text(str(child.pid), encoding="utf-8")
        code = child.wait()
    (job_dir / "exit_code").write_text(str(code), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
