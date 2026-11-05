# -*- coding: utf-8 -*-
"""重新生成 MANIFEST.json 与各模型 CHECKSUMS.json（**基于 LF 规范化后的字节**）。

为什么要规范化：仓库 .gitattributes 把文本统一成 LF，git 里存的 blob 也是 LF；
若直接对 Windows 工作区的 CRLF 字节算哈希，就会与提交字节不符（评审的 29/46 即此问题）。

用法（在包根目录）：
    python build_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_EXT = {".py", ".md", ".json", ".csv", ".txt", ".gitattributes", ".gitignore"}
SKIP_DIRS = {"__pycache__", ".git", ".venv", "recomputed"}


def read_bytes(path: Path) -> bytes:
    b = path.read_bytes()
    return b.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_EXT else b


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.name.endswith(".pyc"):
            continue
        yield p


def main():
    models_dir = HERE / "models"
    for d in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        entries = {}
        for f in sorted(p for p in d.iterdir() if p.is_file() and p.name != "CHECKSUMS.json"):
            entries[f.name] = sha256(read_bytes(f))
        (d / "CHECKSUMS.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[{d.name}] {len(entries)} files")

    manifest_path = HERE / "MANIFEST.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for key in ("package", "purpose", "models", "excluded", "feature_contract", "protocol"):
        if key not in meta:
            raise SystemExit(f"MANIFEST 缺少字段 {key}，请先恢复原清单再重生成")
    files = []
    for f in walk(HERE):
        rel = f.relative_to(HERE).as_posix()
        if rel in ("MANIFEST.json", "self_check.json", "hash_verification.json"):
            continue
        b = read_bytes(f)
        files.append({"path": rel, "bytes": len(b), "sha256": sha256(b)})
    meta["external_dependencies"] = {
        "dataset": "../dataset_same8/train.csv（仅复算 OOF 评分 / 自检时需要；仓库内已跟踪）",
        "scoring_dir": "scoring/（包内自带 surrogate_scoring_v3_2.py + EVALUATION_POLICY_V3_2.json）",
        "split_manifest": "frozen/split_manifest.json（包内只读副本，sdm374.same8.splits.2x5x3.frozen.v1）",
    }
    meta["line_ending"] = "所有文本按 LF 规范化后计算（见 .gitattributes）；校验请用 verify_hashes.py"
    meta["files"] = files
    manifest_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"MANIFEST 收录 {len(files)} 个文件")


if __name__ == "__main__":
    sys.exit(main())
