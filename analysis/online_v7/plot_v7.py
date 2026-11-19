# -*- coding: utf-8 -*-
"""Plot the v7 real run (runs/explore_v7_s7) as R curve, learning curve, step/channel evolution
and weight structure.

    python plot_v7.py            # write to verify_out/viz (git-ignored)
    python plot_v7.py --write    # overwrite the committed viz_explore_v7/

The v4 and h6 traces are shown for reference only; they never feed a proposal.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LOG = HERE / "runs" / "explore_v7_s7" / "online_log.jsonl"
REF = HERE / "runs" / "pr25_repro" / "online_log.jsonl"
REF_V6 = HERE / "runs" / "explore_v6_s7" / "online_log.jsonl"
OUT = HERE / "verify_out" / "viz"
COMMITTED = HERE / "viz_explore_v7"
TRAIN = HERE / "data" / "train.csv"
FREEZE = 11
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
FEATS = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
GROUPS = [("model", "#4C72B0"), ("coarse", "#DD8452"), ("fine", "#55A868"),
          ("block", "#C44E52"), ("restart", "#8172B3"), ("nav", "#937860"),
          ("scan", "#DA8BC3"), ("random", "#7F7F7F")]
GROUP_OF = {"model": "model", "coarse": "coarse", "fine": "fine", "zero": "fine", "tiny": "fine",
            "block": "block", "restart": "restart", "nav": "nav", "scan": "scan",
            "grid": "scan", "icombo": "scan", "scale1": "random", "multi": "random",
            "shift": "random", "fill": "random", "random": "random"}


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path


def load_points(path):
    recs = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    return recs


def build_rows(recs, baseline):
    points = [r for r in recs if r.get("type") == "point"]
    dones = {r["call"]: r for r in recs if r.get("type") == "call_done"}
    by_call = defaultdict(list)
    for p in points:
        by_call[p["call"]].append(p)
    rows, best = [], baseline
    for c in sorted(by_call):
        pts = by_call[c]
        old = best
        best = min(best, min(p["R"] for p in pts))
        base = next((p["R"] for p in pts if p["meta"].get("kind") == "base"), old)
        rows.append({"call": c, "points": pts, "done": dones.get(c, {}), "batch_min": min(p["R"] for p in pts),
                     "batch_med": float(np.median([p["R"] for p in pts])), "best": best,
                     "improvement": old - best, "base_R": base,
                     "hits": sum(p["R"] < base - 1e-9 for p in pts if p["meta"].get("kind") == "single")})
    return rows


def shade(ax, rows):
    calls = [r["call"] for r in rows]
    if FREEZE in calls:
        ax.axvspan(0.5, FREEZE + 0.5, color="#9E9E9E", alpha=0.10, lw=0)
        ax.text((FREEZE + 1) / 2, 0.035, "frozen v4 batches", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#616161")
        ax.text(FREEZE + (calls[-1] - FREEZE) / 2 + 0.5, 0.035, "in-basin multiscale",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color="#1565C0")


def fig_r_curve(rows, baseline, ref_rows, ref_v6_rows=()):
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    x = [r["call"] for r in rows]
    ax.plot(x, [r["batch_min"] for r in rows], "o-", ms=4, lw=1.2,
            color="#4C72B0", label="batch minimum R")
    ax.plot([0] + x, [baseline] + [r["best"] for r in rows], "s-", ms=4, lw=1.9,
            color="#C44E52", label="best R so far (v7)")
    if ref_rows:
        xr = [r["call"] for r in ref_rows]
        ax.plot([0] + xr, [baseline] + [r["best"] for r in ref_rows], "--", lw=1.3,
                color="#7F7F7F", label="PR 25 reference (v4)")
    if ref_v6_rows:
        x6 = [r["call"] for r in ref_v6_rows]
        ax.plot([0] + x6, [baseline] + [r["best"] for r in ref_v6_rows], ":", lw=1.4,
                color="#2E7D32", label="h6 multiscale")
    ax.axhline(102.1433224203, ls=":", lw=1.0, color="#8172B3",
               label="v4 after 35 calls: R = 102.143322")
    best = rows[-1]["best"]
    ax.annotate(f"call {rows[-1]['call']}: best R = {best:.6f}", xy=(x[-1], best),
                xytext=(-150, 95), textcoords="offset points", color="#C44E52", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))
    ax.set_xticks(x)
    ax.set_xlabel("solver call (50 points each)")
    ax.set_ylabel("Ramping R")
    ax.set_title("v7: keep the whole v4 batch, replace zero-yield slots with multiscale candidates")
    ax.grid(alpha=0.3)
    shade(ax, rows)
    ax.legend(fontsize=8, loc="upper right")
    return save(fig, "01_R_curve.png")


def fig_learning(rows, baseline):
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    R, idx, calls = [], [], []
    for r in rows:
        for p in r["points"]:
            idx.append(len(idx) + 1)
            R.append(p["R"])
            calls.append(r["call"])
    idx, R, calls = np.array(idx), np.array(R), np.array(calls)
    sc = ax.scatter(idx, R, s=8, c=calls, cmap="viridis", alpha=0.65)
    ax.step(np.r_[0, idx], np.minimum.accumulate(np.r_[baseline, R]), where="post",
            color="#C44E52", lw=1.8, label="best R so far")
    ax.axhline(baseline, ls=":", color="grey", lw=1.0, label=f"best training row R = {baseline:.4f}")
    if FREEZE:
        ax.axvline(FREEZE * 50 + 0.5, ls="--", color="#1565C0", lw=1.1,
                   label=f"end of frozen phase (call {FREEZE})")
    ax.set_xlabel("cumulative real evaluations")
    ax.set_ylabel("Ramping R")
    ax.set_title(f"Learning curve: {len(idx)} real evaluations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=ax, pad=0.015, label="solver call")
    return save(fig, "02_learning_curve.png")


def fig_steps(rows):
    """上：每轮各移动通道的实际点数；下：步长与停滞状态。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1.0]})
    x = np.array([r["call"] for r in rows])
    comp = Counter()
    series = defaultdict(list)
    for r in rows:
        c = Counter()
        for p in r["points"]:
            m = p["meta"]
            if m.get("kind") in ("base", "combo"):
                c["base/combo"] += 1
                continue
            c[GROUP_OF.get(m.get("move", ""), "random")] += 1
        comp.update(c)
        for g, _ in GROUPS:
            series[g].append(c.get(g, 0))
    bottom = np.zeros(len(x))
    for g, color in GROUPS:
        vals = np.array(series[g])
        if vals.sum() == 0:
            continue
        ax1.bar(x, vals, bottom=bottom, width=0.78, color=color, label=f"{g} ({comp[g]})")
        bottom += vals
    ax1.set_ylabel("points per call")
    ax1.set_title("Channel mix: coarse strides, block restarts and fine steps")
    ax1.legend(fontsize=7, ncol=3, loc="upper right")
    ax1.grid(alpha=0.25, axis="y")

    sig = [r["done"].get("sigma") for r in rows]
    if any(s is not None for s in sig):
        for b, marker in enumerate(["o", "s", "^"]):
            ys = [s[b] if s is not None else np.nan for s in sig]
            ax2.plot(x, ys, marker=marker, ms=4, lw=1.2, label=f"block {b + 1} sigma")
    ax2.axvline(FREEZE + 0.5, ls="--", color="#1565C0", lw=1.0)
    ax2.set_ylabel("log10 sigma")
    ax2.set_xlabel("solver call")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=3, loc="upper left")
    ax2.text(FREEZE + 0.7, ax2.get_ylim()[0] + 0.02, "multiscale from call 12", fontsize=8,
             color="#1565C0")
    fig.tight_layout()
    return save(fig, "03_step_evolution.png")


def fig_structure(rows, baseline):
    inc_W, current = [], None
    for r in rows:
        if current is None:
            current = min((p for rr in rows if rr["call"] <= r["call"] for p in rr["points"]),
                          key=lambda p: p["R"])["W"]
        best_point = min((p for rr in rows if rr["call"] <= r["call"] for p in rr["points"]),
                         key=lambda p: p["R"])
        inc_W.append(np.array(best_point["W"]).reshape(3, 7))
    x = np.array([r["call"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    for b, ax in enumerate(axes):
        bottom = np.zeros(len(x))
        for f in range(7):
            vals = np.array([w[b, f] / w[b].sum() for w in inc_W])
            ax.bar(x, vals, bottom=bottom, width=0.78, color=COLORS[f],
                   label=FEATS[f] if b == 0 else None)
            bottom += vals
        ax.axvline(FREEZE + 0.5, ls="--", color="#1565C0", lw=1.0)
        ax.set_title(f"Building {b + 1}")
        ax.set_xlabel("solver call")
        ax.set_xticks(x[::2])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("normalised weight of the incumbent")
    axes[0].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.suptitle("Structure of the best point over calls (dashed line: start of the multiscale phase)")
    fig.tight_layout()
    return save(fig, "04_structure.png")


def write_tables(rows, baseline):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_call.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["call", "phase", "n", "batch_min", "batch_median", "best_so_far",
                  "improvement", "hits", "sigma_1", "sigma_2", "sigma_3", "solver_seconds"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            sig = r["done"].get("sigma") or ["", "", ""]
            writer.writerow({"call": r["call"], "phase": r["done"].get("phase", ""),
                             "n": len(r["points"]), "batch_min": r["batch_min"],
                             "batch_median": r["batch_med"], "best_so_far": r["best"],
                             "improvement": r["improvement"], "hits": r["hits"],
                             "sigma_1": sig[0], "sigma_2": sig[1], "sigma_3": sig[2],
                             "solver_seconds": r["done"].get("eval_secs", "")})
    best_point = min((p for r in rows for p in r["points"]), key=lambda p: p["R"])
    freeze_best = min((p for r in rows if r["call"] <= FREEZE for p in r["points"]), key=lambda p: p["R"])
    lines = ["v7 hybrid search, real solver results", f"completed calls: {len(rows)} / 20",
             f"best R at the end of the frozen phase (call {FREEZE}): {freeze_best['R']:.10f}",
             f"final best R: {best_point['R']:.10f} (call {best_point['call']}, point {best_point['seq']})",
             f"total drop below the best training row: {baseline - best_point['R']:.10f}",
             f"drop added in the multiscale phase: {freeze_best['R'] - best_point['R']:.10f}",
             f"reaches R<=102.3: {best_point['R'] <= 102.3}",
             f"reaches R<=102.15: {best_point['R'] <= 102.15}",
             f"reaches R<=102.1433224203: {best_point['R'] <= 102.1433224203}", "",
             "per call:"]
    for r in rows:
        lines.append(f"  call {r['call']:2d} [{r['done'].get('phase', ''):18s}] "
                     f"batch_min={r['batch_min']:.8f}  best={r['best']:.8f}  "
                     f"imp={r['improvement']:.8f}  hits={r['hits']}")
    lines += ["", "weight vector of the best point (3x7):",
              np.array2string(np.array(best_point["W"]).reshape(3, 7),
                              precision=8, suppress_small=True, max_line_width=140)]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="overwrite the committed viz_explore_v7/ instead of verify_out/viz/")
    args = ap.parse_args()
    global OUT
    if args.write:
        OUT = COMMITTED
    recs = load_points(LOG)
    with TRAIN.open(newline="", encoding="utf-8") as fh:
        baseline = min(float(r["ramping"]) for r in csv.DictReader(fh))
    rows = build_rows(recs, baseline)
    if not rows:
        raise SystemExit("日志中尚无真实搜索点")
    ref_rows = []
    if REF.exists():
        ref_rows = [r for r in build_rows(load_points(REF), baseline) if r["call"] <= len(rows)]
    ref_v6_rows = []
    if REF_V6.exists():
        ref_v6_rows = [r for r in build_rows(load_points(REF_V6), baseline) if r["call"] <= len(rows)]
    OUT.mkdir(parents=True, exist_ok=True)
    images = [fig_r_curve(rows, baseline, ref_rows, ref_v6_rows), fig_learning(rows, baseline),
              fig_steps(rows), fig_structure(rows, baseline)]
    print(write_tables(rows, baseline))
    for path in images:
        print(path)


if __name__ == "__main__":
    main()
