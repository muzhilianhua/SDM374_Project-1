# -*- coding: utf-8 -*-
"""校验 MANIFEST.json 与四个模型 CHECKSUMS.json。

默认对**提交里的 blob**（git HEAD）逐条比对：`git cat-file blob HEAD:<path>` 取字节，
按 .gitattributes 规则做 LF 规范化后算 SHA256。这样能抓出"工作区哈希对、提交字节不对"
的 CRLF 问题。

用法（在包根目录）：
    python verify_hashes.py                  # 对 HEAD blob 校验（评审要求的默认方式）
    python verify_hashes.py --working-tree   # 对当前工作区校验（用于 git archive 干净检出）
    python verify_hashes.py --rev HEAD~1     # 对指定提交校验

退出码 0 = 全部一致；1 = 有不符（明细逐条打印）。
结果同时写入 self_check.json 的 hash_verification 字段。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_EXT = {".py", ".md", ".json", ".csv", ".txt", ".gitattributes", ".gitignore"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm(path: str, b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n") if Path(path).suffix.lower() in TEXT_EXT else b


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=str(cwd or HERE), check=True,
                          capture_output=True, text=True).stdout


def load_from_head(rev):
    repo = Path(_git("rev-parse", "--show-toplevel").strip())
    rel = HERE.relative_to(repo).as_posix()
    names = [n for n in _git("-c", "core.quotepath=false", "ls-tree", "-r", "--name-only",
                             rev, "--", rel, cwd=repo).splitlines() if n]
    out = {}
    for n in names:
        b = subprocess.run(["git", "cat-file", "blob", f"{rev}:{n}"], cwd=str(repo),
                           check=True, capture_output=True).stdout
        out[n[len(rel) + 1:]] = b
    return out


def load_from_worktree():
    out = {}
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or any(s in p.relative_to(HERE).parts for s in ("__pycache__", ".git", ".venv")):
            continue
        out[p.relative_to(HERE).as_posix()] = p.read_bytes()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--working-tree", action="store_true")
    a = ap.parse_args()

    blobs = load_from_worktree() if a.working_tree else load_from_head(a.rev)
    mode = "working-tree" if a.working_tree else f"git:{a.rev}"
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8-sig"))

    listed = {e["path"]: e for e in manifest["files"]}
    actual = {k for k in blobs
              if k not in ("MANIFEST.json", "self_check.json", "hash_verification.json")}
    missing = sorted(set(listed) - actual)
    extra = sorted(actual - set(listed))
    bad = []
    for path, e in listed.items():
        b = blobs.get(path)
        if b is None:
            continue
        nb = _norm(path, b)
        if _sha(nb) != e["sha256"] or len(nb) != e["bytes"]:
            bad.append(path)

    model_bad, model_missing, model_checked = [], [], 0
    for d in sorted(p for p in (HERE / "models").iterdir() if p.is_dir()):
        cs_path = f"models/{d.name}/CHECKSUMS.json"
        if cs_path not in blobs:
            continue
        cs = json.loads(_norm(cs_path, blobs[cs_path]).decode("utf-8"))
        model_checked += len(cs)
        for name, expected in cs.items():
            p = f"models/{d.name}/{name}"
            if p not in blobs:
                model_missing.append(p)
            elif _sha(_norm(p, blobs[p])) != expected:
                model_bad.append(p)

    report = {"mode": mode, "manifest_files": len(listed),
              "checked_hashes": len(listed) + model_checked,
              "mismatched": len(bad), "missing": len(missing), "extra": len(extra),
              "model_checksums_checked": model_checked,
              "model_mismatched": len(model_bad), "model_missing": len(model_missing),
              "passed": not (bad or missing or extra or model_bad or model_missing)}
    for key in ("mismatched", "missing", "extra", "model_mismatched", "model_missing"):
        if report[key]:
            print(f"{key}:")
            for p in (bad if key == "mismatched" else missing if key == "missing" else extra
                      if key == "extra" else model_bad if key == "model_mismatched" else model_missing):
                print(f"  - {p}")

    print(f"[{mode}] MANIFEST {len(listed)} 条，模型 CHECKSUMS {report['model_checksums_checked']} 条；"
          f"不符 {len(bad)} + 模型 {len(model_bad)}，缺失 {len(missing) + len(model_missing)}，多余 {len(extra)}")

    sc_path = HERE / "self_check.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text(encoding="utf-8-sig"))
    else:
        sc = {}
    sc["hash_verification"] = report
    sc_path.write_text(json.dumps(sc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
