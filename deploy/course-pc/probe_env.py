"""Record the course computer's environment for the online run package (T03, T10).

Standard library only, so it runs under whatever Python the machine has (3.8+), including the bundled
one. It never starts solver.exe and never opens usage.count; it only reports where they are.

    python probe_env.py [--out DIR] [--search-seconds 20]

Writes probe-<host>-<time>.json into DIR (default: ../../../results next to the kit's .bat files)
and prints a short summary.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KIT = REPO.parent
SOLVER_NAMES = ("solver.exe", "RampingSolver.exe")
PINNED = REPO / "solver" / "CHECKSUMS.sha256"


def run(command: list[str], timeout: float = 20.0) -> dict:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        return {"ok": done.returncode == 0, "code": done.returncode,
                "out": done.stdout.strip()[-2000:], "err": done.stderr.strip()[-1000:]}
    except FileNotFoundError:
        return {"ok": False, "missing": True}
    except Exception as error:  # timeouts and anything platform-specific
        return {"ok": False, "error": repr(error)}


def tool(name: str, *args: str) -> dict:
    path = shutil.which(name)
    info = {"path": path}
    if path:
        info.update(run([path, *args]))
    return info


def memory() -> dict:
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_gb": round(status.ullTotalPhys / 2**30, 2), "free_gb": round(status.ullAvailPhys / 2**30, 2)}
        return {}
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return {"total_gb": round(pages * size / 2**30, 2)}
    except (ValueError, OSError, AttributeError):
        return {}


def reach(url: str) -> dict:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=6) as response:
            return {"ok": True, "status": response.status, "seconds": round(time.perf_counter() - start, 2)}
    except Exception as error:
        return {"ok": False, "error": repr(error)[:300]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned() -> str | None:
    try:
        for line in PINNED.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == "solver.exe":
                return parts[0].lower()
    except OSError:
        pass
    return None


def search_roots() -> list[Path]:
    home = Path.home()
    roots = [REPO / "solver", KIT, *list(KIT.parents)[:3], home / "Desktop", home / "Downloads", home / "Documents"]
    if sys.platform == "win32":
        roots += [Path(f"{letter}:\\") for letter in "CDEF" if Path(f"{letter}:\\").exists()]
    seen, unique = set(), []
    for root in roots:
        key = str(root).lower()
        if key not in seen and root.exists():
            seen.add(key)
            unique.append(root)
    return unique


def find_solvers(budget: float) -> list[dict]:
    deadline = time.monotonic() + budget
    skip = {"windows", "program files", "program files (x86)", "programdata", "$recycle.bin", "appdata",
            "node_modules", ".git", "site-packages", "system volume information"}
    found: dict[str, Path] = {}
    for root in search_roots():
        stack = [(root, 0)]
        while stack and time.monotonic() < deadline:
            folder, depth = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_file() and entry.name in SOLVER_NAMES:
                        found.setdefault(os.path.normcase(entry.path), Path(entry.path))
                    elif entry.is_dir(follow_symlinks=False) and depth < 4 and entry.name.lower() not in skip:
                        stack.append((Path(entry.path), depth + 1))
                except OSError:
                    continue
    expected = pinned()
    results = []
    for path in found.values():
        usage = path.parent / "usage.count"
        item = {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
        item["matches_pinned"] = expected is not None and item["sha256"] == expected
        item["usage_count_present"] = usage.exists()  # never read or changed
        if usage.exists():
            stat = usage.stat()
            item["usage_count_size"] = stat.st_size
            item["usage_count_mtime"] = dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        results.append(item)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=KIT / "results")
    parser.add_argument("--search-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.perf_counter()
    report: dict = {"time": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname()}
    report["os"] = {"platform": platform.platform(), "release": platform.release(), "version": platform.version(),
                    "machine": platform.machine(), "processor": platform.processor()}
    report["cpu_count"] = os.cpu_count()
    report["memory"] = memory()
    report["python_running_this"] = {"executable": sys.executable, "version": sys.version}
    report["kit"] = {"path": str(KIT), "non_ascii": not str(KIT).isascii(), "has_space": " " in str(KIT),
                     "version_file": (REPO / "VERSION").read_text(encoding="utf-8").strip()
                     if (REPO / "VERSION").exists() else None}
    usage = shutil.disk_usage(KIT)
    report["disk_free_gb"] = round(usage.free / 2**30, 1)
    try:
        with tempfile.NamedTemporaryFile(dir=KIT, delete=True):
            report["kit_writable"] = True
    except OSError as error:
        report["kit_writable"] = repr(error)

    report["tools"] = {
        "python_on_path": tool("python", "--version"),
        "python3_on_path": tool("python3", "--version"),
        "py_launcher": tool("py", "-0p"),
        "uv": tool("uv", "--version"),
        "conda": tool("conda", "--version"),
        "pip": tool("pip", "--version"),
        "git": tool("git", "--version"),
        "tar": tool("tar", "--version"),
    }
    if sys.platform == "win32":
        report["tools"]["where_python"] = run(["where", "python"])
        report["powershell_execution_policy"] = run(
            ["powershell", "-NoProfile", "-Command", "Get-ExecutionPolicy -List | Out-String"])
        report["codepage"] = run(["cmd", "/c", "chcp"])
    bundled = KIT / "python" / "python.exe"
    report["bundled_python"] = run([str(bundled), "--version"]) if bundled.exists() else {"present": False}
    report["network"] = {"pypi": reach("https://pypi.org/simple/pip/"),
                         "github": reach("https://github.com"),
                         "astral": reach("https://astral.sh")}
    report["env"] = {key: os.environ.get(key) for key in
                     ("USERNAME", "COMPUTERNAME", "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS",
                      "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PYTHONPATH", "OMP_NUM_THREADS")}
    report["solvers"] = find_solvers(args.search_seconds)
    report["probe_seconds"] = round(time.perf_counter() - started, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = args.out / f"probe-{report['host']}-{stamp}.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    tools = report["tools"]
    lines = [
        f"OS            : {report['os']['platform']}",
        f"CPU / memory  : {report['cpu_count']} threads, {report['memory'].get('total_gb')} GB",
        f"Python (PATH) : {tools['python_on_path'].get('out') or tools['python_on_path'].get('err') or 'not found'}",
        f"py launcher   : {(tools['py_launcher'].get('out') or 'not found').replace(os.linesep, ' | ')}",
        f"uv / conda    : {tools['uv'].get('out') or 'no uv'} / {tools['conda'].get('out') or 'no conda'}",
        f"git           : {tools['git'].get('out') or 'no git'}",
        f"Internet      : pypi {report['network']['pypi']['ok']}, github {report['network']['github']['ok']}",
        f"Kit path      : {KIT} (non-ASCII: {report['kit']['non_ascii']}, spaces: {report['kit']['has_space']})",
        f"Disk free     : {report['disk_free_gb']} GB",
        f"solver.exe    : {len(report['solvers'])} found",
    ]
    for item in report["solvers"]:
        lines.append(f"  - {item['path']}  pinned={item['matches_pinned']}  usage.count={item['usage_count_present']}")
    print("\n".join(lines))
    print(f"\nWrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
