# -*- coding: utf-8 -*-
"""v7 新探索：前 11 轮冻结为 v4 原样，第 12–20 轮保留 v4 完整批次并换入多尺度候选。

用法：
    python explore_v7.py verify-freeze   # 不调用求解器，逐点核对 1–11 轮与参考运行一致
    python explore_v7.py dry-run         # 不调用求解器，检查第 12 轮替换结果
    python explore_v7.py run             # 真实运行 20 轮，中断后同命令续跑
    python explore_v7.py audit           # 重建审计报告

额度：使用全新的独立求解器副本 explore_v7_solver（复制得到，首次运行自动生成新的
usage.count），只消耗 20 次，恰好等于 20 轮；不读取、不修改、不删除任何已有 usage.count。
求解器链路正确性沿用同一 sha256 副本的既有证据：两次均匀权重调用返回完全相同的
R=156.2456085834（见 runs/accelerated15_s7/chain_check.jsonl），因此不在本次额度内重复对拍。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "policy"))

from online_run import (BATCH, PINNED, SEED, SOLVER_SHA256, State, load_training,  # noqa: E402
                        norm_blocks, propose, sha256)
from policy_v7 import BlockDescentPolicyV7, ParamsV7  # noqa: E402

CALLS = 20
FREEZE = 11
OUT = HERE / "runs" / "explore_v7_s7"
LOG = OUT / "online_log.jsonl"
CONFIG = OUT / "run_config.json"
SUMMARY = OUT / "summary.json"
AUDIT = OUT / "audit_report.json"
FREEZE_CHECK = OUT / "freeze_check.json"
SOLVER_DIR = HERE / "explore_v7_solver"
REFERENCE = HERE / "runs" / "pr25_repro" / "online_log.jsonl"
CHAIN_EVIDENCE = HERE / "runs" / "accelerated15_s7" / "chain_check.jsonl"
POLICY_FILES = [HERE / "policy" / n for n in
                ("policy.py", "policy_v4.py", "policy_v5.py", "policy_v6.py", "policy_v7.py",
                 "surrogate.py")]
MIN_FREE_BYTES = int(1.5 * 1024 ** 3)
_AUDIT_PHASE = "startup"
_FILE_ACCESS = []


def make_policy():
    return BlockDescentPolicyV7(ParamsV7(shift_bias=True, n_nav=3), seed=SEED)


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, records):
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(canonical(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def ensure_disk_space():
    free = shutil.disk_usage(Path(tempfile.gettempdir()).anchor).free
    if free >= MIN_FREE_BYTES:
        return free
    now = time.time()
    for path in Path(tempfile.gettempdir()).glob("_MEI*"):
        try:
            if path.is_dir() and now - path.stat().st_mtime > 86400:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
    raise RuntimeError(f"系统盘可用空间仅 {free / 1024 ** 3:.2f} GB，已停止新批次")


def prepare_solver() -> Path:
    """全新独立副本；若目标目录已有任何运行痕迹则拒绝启动。"""
    src = HERE / "solver" / "solver.exe"
    if sha256(src) != SOLVER_SHA256:
        raise SystemExit("源 solver.exe 与课程固定版本不一致")
    SOLVER_DIR.mkdir(parents=True, exist_ok=True)
    dst = SOLVER_DIR / "solver.exe"
    if not dst.exists():
        if any(SOLVER_DIR.iterdir()):
            raise SystemExit("目标目录非空且无求解器，拒绝覆盖未知状态")
        shutil.copy2(src, dst)
    if sha256(dst) != SOLVER_SHA256:
        raise SystemExit("新副本 solver.exe 校验失败")
    return dst


def environment_info():
    versions = {k: importlib.metadata.version(k) for k in
                ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    return {"python": platform.python_version(), "machine": platform.platform(),
            "logical_cpus": os.cpu_count(), "packages": versions,
            "thread_env": {k: os.environ.get(k) for k in
                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "thread_note": "未强制单线程，以与参考运行 pr25_repro 的默认线程环境一致"}


def build_config():
    return {
        "policy": "v7-hybrid", "seed": SEED, "calls": CALLS, "batch": BATCH,
        "freeze_calls": FREEZE, "max_solver_calls": CALLS,
        "training": "data/train.csv", "training_sha256": sha256(HERE / "data" / "train.csv"),
        "training_rows": 2048, "solver_sha256": SOLVER_SHA256,
        "policy_sha256": {p.name: sha256(p) for p in POLICY_FILES},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "requirements_sha256": sha256(HERE / "requirements.txt"),
        "history_used_for_policy": False, "rng_uses_clock": False,
        "environment": environment_info(),
    }


def write_config_once():
    expected = canonical(build_config()) + "\n"
    if CONFIG.exists():
        if CONFIG.read_text(encoding="utf-8") != expected:
            raise SystemExit("现有 run_config.json 与当前代码或环境不一致，拒绝混合续跑")
    else:
        CONFIG.write_text(expected, encoding="utf-8")


def solver_payload(batch):
    text = json.dumps({"weights": np.asarray(batch).tolist()}, separators=(",", ":"))
    return text, digest_text(text)


def run_solver_raw(exe: Path, batch, timeout=3600.0):
    payload, payload_sha = solver_payload(batch)
    proc = subprocess.run([str(exe)], input=payload, capture_output=True, text=True,
                          cwd=str(exe.parent), timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise RuntimeError(f"solver exit {proc.returncode}: {proc.stderr.strip()[:500]}")
    values = json.loads(proc.stdout)["R"]
    R = [float(v) for v in values]
    if len(R) != len(batch) or not all(np.isfinite(R)):
        raise RuntimeError(f"solver 返回 {len(R)} 个有效值，预期 {len(batch)}")
    return R, payload_sha, digest_text(proc.stdout)


def load_reference():
    recs = [json.loads(x) for x in REFERENCE.open(encoding="utf-8") if x.strip()]
    calls = {}
    for r in recs:
        if r.get("type") == "point":
            calls.setdefault(r["call"], []).append(r)
    return [calls[c] for c in sorted(calls)]


def validate_batch(batch, X_train, sent):
    w = np.asarray(batch, float)
    if w.shape != (BATCH, 21):
        raise ValueError(f"候选形状 {w.shape}，预期 {(BATCH, 21)}")
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError("候选含非有限值或负权重")
    sums = w.reshape(-1, 3, 7).sum(axis=2)
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError("存在楼栋权重和不为 1 的候选")
    known = [norm_blocks(X_train).reshape(-1, 21)]
    if len(sent):
        known.append(np.asarray(sent).reshape(-1, 21))
    known.append(w)
    for i, row in enumerate(w):
        for j, rows in enumerate(known):
            compare = rows[:i] if j == len(known) - 1 else rows
            if len(compare) and np.min(np.abs(compare - row).max(axis=1)) <= 1e-9:
                raise ValueError(f"候选 {i} 与训练点、已测点或批内前序点重复")
    return {"n": len(w), "shape": list(w.shape), "finite_nonnegative": True,
            "building_sums_one": True, "no_training_or_sent_or_batch_duplicate": True,
            "exact_zero_weights": int((w == 0).sum()),
            "max_building_sum_error": float(np.abs(sums - 1).max())}


def propose_checked(policy, st, X):
    batch, metas, info = propose(policy, st, X)
    sent = np.array([e["W"].reshape(21) for c in st.calls for e in c]).reshape(-1, 21)
    validation = validate_batch(batch, X, sent)
    if len(st.calls) >= CALLS:
        raise ValueError("20 轮搜索预算已用完")
    return batch, metas, info, validation


def audit_hook(event, args):
    if event == "open" and _AUDIT_PHASE == "propose":
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            _FILE_ACCESS.append({"phase": "propose", "path": os.fspath(path), "mode": args[1]})


def cmd_verify_freeze():
    """用参考运行自己的历史重放 1–11 轮，按多重集合核对 v6 冻结阶段是否完全一致。

    参考日志中第 3 轮的点写入顺序与提交顺序不同（集合相同），因此这里比较集合而非行序。
    """
    X, _ = load_training()
    ref = load_reference()
    st = State(X, load_training()[1])
    rows, ok = [], True
    for c in range(1, FREEZE + 1):
        batch, metas, _ = propose(make_policy(), st, X)
        ref_batch = np.array([e["W"] for e in ref[c - 1]])
        d = np.abs(batch[:, None, :] - ref_batch[None, :, :]).max(axis=2)
        diff = float(d.min(axis=1).max())
        matched = int((d.min(axis=1) <= 1e-12).sum())
        row = {"call": c, "n": len(batch), "matched_rows": matched, "max_abs_diff": diff,
               "multiset_identical": matched == len(batch) == len(ref_batch),
               "pass": matched == len(batch) == len(ref_batch) and diff <= 1e-12}
        ok &= row["pass"]
        rows.append(row)
        print(f"  call {c:2d}: 匹配 {matched}/{len(batch)}  max|dW|={diff:.2e}  "
              f"{'PASS' if row['pass'] else 'FAIL'}", flush=True)
        st.calls.append([{"W": np.array(e["W"]).reshape(3, 7), "R": e["R"], "meta": e["meta"]}
                         for e in ref[c - 1]])
    OUT.mkdir(parents=True, exist_ok=True)
    FREEZE_CHECK.write_text(json.dumps({"freeze_calls": FREEZE, "all_pass": ok,
                                        "comparison": "multiset", "per_call": rows},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"冻结校验：{'PASS' if ok else 'FAIL'}（未调用 solver）")
    if not ok:
        raise SystemExit("冻结阶段与参考运行不一致，停止")


def cmd_dry_run():
    X, y = load_training()
    ref = load_reference()
    st = State(X, y)
    for c in range(1, FREEZE + 1):
        st.calls.append([{"W": np.array(e["W"]).reshape(3, 7), "R": e["R"], "meta": e["meta"]}
                         for e in ref[c - 1]])
    batch1, meta1, info1, validation = propose_checked(make_policy(), st, X)
    batch2, meta2, _, _ = propose_checked(make_policy(), st, X)
    exact = np.array_equal(batch1, batch2) and canonical(meta1) == canonical(meta2)
    moves = {}
    for m in meta1:
        moves[m.get("move", m.get("kind"))] = moves.get(m.get("move", m.get("kind")), 0) + 1
    print(canonical({"calls": 0, "solver_calls": 0, "batch": validation, "replay_exact": exact,
                     "R_inc": info1["R_inc"], "phase": info1["phase"],
                     "stagnant": info1["stagnant"], "sigma": info1["sigma"], "moves": moves}))
    if not exact:
        raise SystemExit("第 12 轮提案重放不一致")
    print("dry-run: PASS（未调用 solver）")


def load_own_state(X, y):
    st = State(X, y)
    if not LOG.exists():
        return st
    recs = [json.loads(x) for x in LOG.open(encoding="utf-8") if x.strip()]
    started = {r["call"] for r in recs if r.get("type") == "call_start"}
    done = {r["call"] for r in recs if r.get("type") == "call_done"}
    if started - done:
        raise SystemExit(f"存在未完成的调用 {sorted(started - done)}；可能已扣额度，需人工检查")
    if sorted(done) != list(range(1, len(done) + 1)):
        raise SystemExit("日志调用编号不连续，拒绝续跑")
    for c in sorted(done):
        pts = sorted((r for r in recs if r.get("type") == "point" and r["call"] == c),
                     key=lambda r: r["seq"])
        st.calls.append([{"W": np.array(r["W"]).reshape(3, 7), "R": r["R"], "meta": r["meta"]}
                         for r in pts])
    return st


def cmd_run():
    global _AUDIT_PHASE
    sys.addaudithook(audit_hook)
    OUT.mkdir(parents=True, exist_ok=True)
    exe = prepare_solver()
    X, y = load_training()
    st = load_own_state(X, y)
    write_config_once()
    if not LOG.exists():
        append_jsonl(LOG, [{"type": "run_info", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **build_config()}])
    else:
        append_jsonl(LOG, [{"type": "resume_info", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "completed_calls": len(st.calls)}])
    best = min([float(y.min())] + [e["R"] for c in st.calls for e in c])
    while len(st.calls) < CALLS:
        c = len(st.calls) + 1
        free_before = ensure_disk_space()
        access_start = len(_FILE_ACCESS)
        _AUDIT_PHASE = "propose"
        t0 = time.time()
        batch, metas, info, validation = propose_checked(make_policy(), st, X)
        t1 = time.time()
        replay, replay_meta, _, _ = propose_checked(make_policy(), st, X)
        replay_exact = np.array_equal(batch, replay) and canonical(metas) == canonical(replay_meta)
        _AUDIT_PHASE = "solver"
        t2 = time.time()
        if not replay_exact:
            raise RuntimeError(f"第 {c} 轮提案重放不一致，未调用 solver")
        _, payload_sha = solver_payload(batch)
        append_jsonl(LOG, [{"type": "call_start", "call": c, "n": len(batch), "budget_ok": True,
                            "validation": validation, "replay_exact": replay_exact,
                            "payload_sha256": payload_sha, "disk_free_gb": free_before / 1024 ** 3,
                            "phase": info.get("phase")}])
        R, payload_sha, response_sha = run_solver_raw(exe, batch)
        t3 = time.time()
        free_after = ensure_disk_space()
        old_best = best
        best = min(best, min(R))
        points = [{"type": "point", "call": c, "seq": i, "W": w.tolist(), "R": r,
                   "meta": m, "payload_sha256": payload_sha, "response_sha256": response_sha}
                  for i, (w, r, m) in enumerate(zip(batch, R, metas))]
        done = {"type": "call_done", "call": c, "n": len(R), "batch_min": min(R),
                "best_so_far": best, "propose_secs": round(t1 - t0, 1),
                "replay_secs": round(t2 - t1, 1), "eval_secs": round(t3 - t2, 1),
                "replay_exact": replay_exact, "payload_sha256": payload_sha,
                "response_sha256": response_sha, "results_in_submission_order": True,
                "disk_free_after_gb": free_after / 1024 ** 3,
                "policy_file_access": _FILE_ACCESS[access_start:], **info}
        append_jsonl(LOG, points + [done])
        st.calls.append([{"W": w.reshape(3, 7), "R": r, "meta": m}
                         for w, r, m in zip(batch, R, metas)])
        marker = "  <-- 新低" if best < old_best - 1e-12 else ""
        print(f"call {c:2d}/20: batch min {min(R):.6f}  best {best:.10f}{marker}"
              f" | propose {t1 - t0:.0f}s replay {t2 - t1:.0f}s solver {t3 - t2:.0f}s", flush=True)
    _AUDIT_PHASE = "post_run_audit"
    write_summary()
    report = write_audit()
    print(f"完成：20 轮真实搜索最优 R={min(e['R'] for c in st.calls for e in c):.10f}")
    print(f"审计：{'PASS' if report['overall_pass'] else 'FAIL'} ({AUDIT})")


def write_summary():
    recs = [json.loads(x) for x in LOG.open(encoding="utf-8") if x.strip()]
    pts = [r for r in recs if r.get("type") == "point"]
    if not pts:
        raise SystemExit("尚无真实搜索点")
    _, y_train = load_training()
    baseline = float(y_train.min())
    best = min(pts, key=lambda r: r["R"])
    by_call = []
    for c in sorted({r["call"] for r in pts}):
        rows = [r for r in pts if r["call"] == c]
        previous = [r["R"] for r in pts if r["call"] < c]
        incumbent = min(previous) if previous else baseline
        batch_min = min(r["R"] for r in rows)
        by_call.append({"call": c, "n": len(rows), "batch_min": batch_min,
                        "best_so_far": min(incumbent, batch_min),
                        "improvement": max(0.0, incumbent - batch_min),
                        "phase": "freeze" if c <= FREEZE else "in_basin_multiscale"})
    summary = {"seed": SEED, "policy": "v6-multiscale", "completed_calls": len(by_call),
               "evaluated_points": len(pts), "best_R": best["R"], "best_call": best["call"],
               "best_seq": best["seq"], "best_W": best["W"], "baseline_train_min": baseline,
               "target_102_3": best["R"] <= 102.3,
               "target_102_15": best["R"] <= 102.15,
               "target_102_143322": best["R"] <= 102.1433224203, "per_call": by_call}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def static_policy_audit():
    findings = []
    forbidden = ("runs/pr25", "runs/from_call20", "103.275294", "102.143322", "reference/")
    io_names = {"open", "read_csv", "read_json", "loadtxt", "genfromtxt"}
    for path in POLICY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in io_names:
                findings.append({"file": path.name, "line": node.lineno, "issue": f"文件读取 {node.func.id}"})
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in forbidden:
                    if token in node.value:
                        findings.append({"file": path.name, "line": node.lineno, "issue": f"外部引用 {token}"})
    return {"pass": not findings, "findings": findings,
            "files_sha256": {p.name: sha256(p) for p in POLICY_FILES}}


def write_audit():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    recs = [json.loads(x) for x in LOG.open(encoding="utf-8") if x.strip()]
    starts = [r for r in recs if r.get("type") == "call_start"]
    dones = [r for r in recs if r.get("type") == "call_done"]
    pts = [r for r in recs if r.get("type") == "point"]
    frozen = [d for d in dones if d["call"] <= FREEZE]
    in_basin = [d for d in dones if d["call"] > FREEZE]
    forbidden_access = []
    for done in dones:
        for item in done.get("policy_file_access", []):
            low = item["path"].lower().replace("\\", "/")
            if any(t in low for t in ("runs/", "reference/", ".csv", ".jsonl")):
                forbidden_access.append({"call": done["call"], **item})
    env = config["environment"]
    env_ok = (env["python"].startswith(PINNED["python"]) and
              all(env["packages"][k] == PINNED[k] for k in ("numpy", "scipy", "scikit-learn")))
    order_ok = True
    for done in dones:
        rows = sorted((p for p in pts if p["call"] == done["call"]), key=lambda p: p["seq"])
        order_ok &= [p["seq"] for p in rows] == list(range(done["n"]))
        order_ok &= all(p["response_sha256"] == done["response_sha256"] for p in rows)
        order_ok &= all(p["payload_sha256"] == next(s for s in starts if s["call"] == done["call"])["payload_sha256"]
                        for p in rows)
    freeze_ok = FREEZE_CHECK.exists() and json.loads(FREEZE_CHECK.read_text(encoding="utf-8"))["all_pass"]
    chain = []
    if CHAIN_EVIDENCE.exists():
        chain = [json.loads(x) for x in CHAIN_EVIDENCE.open(encoding="utf-8") if x.strip()
                 and json.loads(x).get("type") == "chain_point"]
    chain_ok = (len(chain) == 2 and chain[0]["R"] == chain[1]["R"]
                and chain[0]["payload_sha256"] == chain[1]["payload_sha256"]
                and chain[0]["response_sha256"] == chain[1]["response_sha256"]
                and abs(chain[0]["R"] - 156.24560858344194) <= 1e-9)
    checks = {
        "1_training_source": {"pass": config["training_sha256"] == sha256(HERE / "data" / "train.csv")
                                      and config["training_rows"] == 2048,
                               "sha256": config["training_sha256"]},
        "2_policy_no_external_answers": static_policy_audit(),
        "3_runtime_file_access": {"pass": not forbidden_access, "forbidden_access": forbidden_access,
                                   "note": "audit hook 覆盖提案与重放阶段；求解器只接收 JSON 权重"},
        "4_proposal_replay": {"pass": len(dones) > 0 and all(r["replay_exact"] for r in dones),
                               "calls_checked": len(dones)},
        "5_no_reused_points": {"pass": all(r["validation"]["no_training_or_sent_or_batch_duplicate"] for r in starts),
                                "calls_checked": len(starts)},
        "6_result_attribution_order": {"pass": order_ok, "points": len(pts)},
        "7_seed_config_fixed": {"pass": config["seed"] == SEED and config["calls"] == CALLS
                                         and config["batch"] == 50 and not config["rng_uses_clock"],
                                 "seed": config["seed"], "calls": config["calls"]},
        "8_environment_rebuildable": {"pass": env_ok and config["requirements_sha256"] == sha256(HERE / "requirements.txt"),
                                       "environment": env},
        "9_freeze_and_solver": {"pass": freeze_ok and chain_ok,
                                 "freeze_calls": FREEZE, "freeze_check": str(FREEZE_CHECK),
                                 "solver_sha256": config["solver_sha256"],
                                 "fresh_copy_calls_used": len(dones),
                                 "chain_reused_evidence": chain_ok},
        "10_budget": {"pass": len(dones) <= CALLS and all(s["budget_ok"] for s in starts),
                       "search_calls": len(dones), "max_calls": CALLS},
    }
    complete = len(dones) == CALLS
    report = {"complete_20_calls": complete, "freeze_calls": len(frozen),
              "in_basin_calls": len(in_basin), "checks": checks,
              "overall_pass": complete and all(c["pass"] for c in checks.values())}
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("verify-freeze", "dry-run", "run", "audit"))
    args = ap.parse_args()
    if args.cmd == "verify-freeze":
        cmd_verify_freeze()
    elif args.cmd == "dry-run":
        cmd_dry_run()
    elif args.cmd == "run":
        cmd_run()
    else:
        report = write_audit()
        print(canonical({"complete": report["complete_20_calls"],
                         "overall_pass": report["overall_pass"], "report": str(AUDIT)}))


if __name__ == "__main__":
    main()
