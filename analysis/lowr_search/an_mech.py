"""Analysis and figures for the mechanism experiments E7-E9.

    python an_mech.py            # text summary; points not evaluated yet are skipped
    python an_mech.py --plot     # also writes mech_rscan.png, mech_dose.png, mech_pair.png

It rebuilds each experiment's design with the same expressions as the exp_*.py scripts and looks the
points up in eval_log.jsonl, so it can run while an experiment is still in progress.
"""
import json
import sys
import warnings

import numpy as np

from common import MINOR, NAMES, block, to_ratio
from pool import LOG as LOG_FILE, key_of, normalize

warnings.filterwarnings("ignore", category=RuntimeWarning)      # all-NaN windows while a run is in progress
np.set_printoptions(precision=3, suppress=True, linewidth=220)
state = json.load(open("fms_state_run4.json", encoding="utf-8"))
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
MN = [NAMES[k] for k in MINOR]                       # cost, carbon, peak, discomfort, resilience
cache = {}
for line in open(LOG_FILE, encoding="utf-8"):
    rec = json.loads(line)
    cache[key_of(np.asarray(rec["weights"], dtype=float))] = rec["R"]


def delta(W):
    return cache.get(key_of(normalize(W.ravel())[0]), np.nan) - inc_R


def with_block(b, blk):
    W = inc.copy(); W[b] = blk
    return delta(W)


def roll(x, w, fn=np.nanmedian):
    h = w // 2
    return np.array([fn(x[max(0, i - h): i + h + 1]) if np.isfinite(x[max(0, i - h): i + h + 1]).any() else np.nan
                     for i in range(len(x))])


# ------------------------------------------------------------------------------------------------- E9
GRID = np.round(np.arange(0.15, 0.5001, 0.002), 4)


def rscan():
    res = {}
    for b in range(3):
        r0, m0 = to_ratio(inc[b])
        for variant, m in (("minor", m0), ("pure", np.zeros(5))):
            res[(b, variant)] = np.array([with_block(b, block(float(r), m)) for r in GRID])
    return res


def report_rscan(res):
    print("=" * 100 + "\nE9  fine scan of r (step 0.002); delta vs incumbent; 'med' = running median over 11 points (0.02 in r)")
    for b in range(3):
        r0 = to_ratio(inc[b])[0]
        for variant in ("minor", "pure"):
            d = res[(b, variant)]; ok = np.isfinite(d)
            if ok.sum() < 20:
                print(f"  b{b + 1} {variant}: only {int(ok.sum())} points so far"); continue
            med = roll(d, 11); q10 = roll(d, 21, lambda v: np.nanquantile(v, 0.1))
            i = int(np.nanargmin(med)); j = int(np.nanargmin(d))
            after = np.arange(len(GRID)) > i
            on1 = GRID[after & (med > med[i] + 1.0)]; on3 = GRID[after & (med > med[i] + 3.0)]
            basin = (GRID >= 0.20) & (GRID <= 0.30) & ok
            res_b = (d - med)[basin]
            ac = [np.corrcoef(res_b[:-k], res_b[k:])[0, 1] for k in (1, 2, 5)] if basin.sum() > 12 else [np.nan] * 3
            print(f"  b{b + 1} {variant:5s} n={int(ok.sum()):3d}  incumbent r {r0:.4f} | lowest point {d[j]:+.3f} at r={GRID[j]:.3f} | "
                  f"median lowest {med[i]:+.2f} at r={GRID[i]:.3f}; +1 above it from r={on1[0] if len(on1) else np.nan:.3f}, +3 from r={on3[0] if len(on3) else np.nan:.3f} | "
                  f"r in [0.20,0.30]: median {np.nanmedian(d[basin]):+.2f}, q10 {np.nanquantile(d[basin], 0.1):+.2f}, scatter SD {np.nanstd(res_b):.2f}, "
                  f"autocorr lag 1/2/5 {ac[0]:+.2f}/{ac[1]:+.2f}/{ac[2]:+.2f}")
        a, p = res[(b, "minor")], res[(b, "pure")]
        print(f"     value of the minor weights (pure - minor), median by r band: " + "  ".join(
            f"[{lo:.2f},{hi:.2f}) {np.nanmedian((p - a)[(GRID >= lo) & (GRID < hi)]):+.2f}" for lo, hi in ((0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50))))
        print("     running median, minor: " + "  ".join(f"{r:.2f}:{v:+.2f}" for r, v in zip(GRID[::10], roll(a, 11)[::10])))
        print("     running median, pure : " + "  ".join(f"{r:.2f}:{v:+.2f}" for r, v in zip(GRID[::10], roll(p, 11)[::10])))


# ------------------------------------------------------------------------------------------------- E8
LOGG = 10 ** np.linspace(-4, 0, 25)
NEAR = np.array([0.7, 0.85, 0.95, 1.05, 1.2, 1.5])
FACT = np.array([0, 1 / 8, 1 / 4, 1 / 2, 2 ** -0.5, 1, 2 ** 0.5, 2, 4, 8])
PAIRS = {0: (1, 3), 1: (4, 1), 2: (2, 4)}


def dose():
    loo, add, pair = {}, {}, {}
    for b in range(3):
        r0, m0 = to_ratio(inc[b])
        for j in range(5):
            grid = np.unique(np.concatenate([[0.0], LOGG, m0[j] * NEAR if m0[j] > 0 else []]))
            d = []
            for v in grid:
                m = m0.copy(); m[j] = v; d.append(with_block(b, block(r0, m)))
            loo[(b, j)] = (grid, np.array(d))
            d = []
            for v in LOGG:
                m = np.zeros(5); m[j] = v; d.append(with_block(b, block(r0, m)))
            add[(b, j)] = (LOGG, np.array(d))
        j1, j2 = PAIRS[b]
        M = np.full((10, 10), np.nan)
        for a, f1 in enumerate(FACT):
            for c, f2 in enumerate(FACT):
                m = m0.copy(); m[j1] = m0[j1] * f1; m[j2] = m0[j2] * f2
                M[a, c] = with_block(b, block(r0, m))
        pair[b] = M
    return loo, add, pair


def report_dose(loo, add, pair):
    print("=" * 100 + "\nE8  minor weights; m = weight / (degradation + ramping); delta vs incumbent")
    print("\n  [loo] one minor weight varied, the others as in the incumbent")
    print("  bld objective    m_inc      at 0     best (at m)        running-median profile over the log grid 1e-4 .. 1 (every 3rd point)")
    for b in range(3):
        m0 = to_ratio(inc[b])[1]
        for j in range(5):
            g, d = loo[(b, j)]
            if np.isfinite(d).sum() < 5:
                print(f"  b{b + 1}  {MN[j]:11s} not evaluated yet"); continue
            i = int(np.nanargmin(d)); pos = g > 0
            med = roll(d[pos], 5)
            sel = np.searchsorted(g[pos], LOGG[::3] * 0.999)
            prof = "  ".join(f"{g[pos][min(s, pos.sum() - 1)]:.0e}:{med[min(s, pos.sum() - 1)]:+.2f}" for s in sel)
            print(f"  b{b + 1}  {MN[j]:11s} {m0[j]:8.5f}  {d[0]:+7.3f}  {d[i]:+7.3f} ({g[i]:.4g})   {prof}")
    print("\n  [add] the block without minor weights plus ONE minor weight (log grid 1e-4 .. 1)")
    print("  bld  no minors   " + "".join(f"{n:>22s}" for n in MN) + "     (best delta and the m where it occurs)")
    for b in range(3):
        base = with_block(b, block(to_ratio(inc[b])[0], np.zeros(5)))
        cells = []
        for j in range(5):
            g, d = add[(b, j)]
            if np.isfinite(d).sum() < 5:
                cells.append(f"{'n/a':>22s}"); continue
            i = int(np.nanargmin(d)); cells.append(f"{d[i]:+9.3f} ({g[i]:8.1e})  ")
        print(f"  b{b + 1}   {base:+8.3f}   " + "".join(cells))
        cells = []
        for j in range(5):
            g, d = add[(b, j)]
            if np.isfinite(d).sum() < 5:
                cells.append(f"{'':>22s}"); continue
            better = np.nanmean(d < base - 1e-9); cells.append(f"{np.nanmedian(d[(g > 1e-3) & (g < 0.1)]):+9.3f} / {better:4.0%}       ")
        print(f"       median for 1e-3 < m < 0.1 / share of the grid below 'no minors':" + "".join(cells))
    print("\n  [pair] 10 x 10 grid over the two most valuable minor weights (factors 0, 1/8 .. 8 of the incumbent values)")
    for b in range(3):
        j1, j2 = PAIRS[b]; M = pair[b]
        if np.isfinite(M).sum() < 50:
            print(f"  b{b + 1}: {int(np.isfinite(M).sum())} of 100 cells so far"); continue
        b0, b1, b2, b12 = M[0, 0], np.nanmin(M[1:, 0]), np.nanmin(M[0, 1:]), np.nanmin(M[1:, 1:])
        core = M[3:8, 3:8]                              # factors 1/2 .. 2 on both
        print(f"  b{b + 1}  rows {MN[j1]} x columns {MN[j2]}: neither {b0:+.3f} | only {MN[j1]} (best of 9) {b1:+.3f} | only {MN[j2]} (best of 9) {b2:+.3f} | both (best of 81) {b12:+.3f}")
        print(f"       medians: only {MN[j1]} {np.nanmedian(M[1:, 0]):+.2f}, only {MN[j2]} {np.nanmedian(M[0, 1:]):+.2f}, both within 1/2..2 of the incumbent {np.nanmedian(core):+.2f}, both anywhere {np.nanmedian(M[1:, 1:]):+.2f}")
        print("       factor   " + "".join(f"{f:7.2f}" for f in FACT))
        for a, f in enumerate(FACT):
            print(f"       {f:6.2f}   " + "".join(f"{v:+7.2f}" for v in M[a]))


# ------------------------------------------------------------------------------------------------ plots
def plots(res, loo, add, pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    SURF, INK, INK2, MUTED, GRIDC, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    have = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Microsoft YaHei", "SimHei", "Segoe UI", "DejaVu Sans") if f in have]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fonts, "axes.unicode_minus": False, "font.size": 9,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })
    zh = "Microsoft YaHei" in have or "SimHei" in have
    T = (lambda c, e: c) if zh else (lambda c, e: e)
    bname = [T(f"{b + 1} 号楼", f"Building {b + 1}") for b in range(3)]

    # --- r scan ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    YMAX = 8.0
    for b in range(3):
        a = ax[b]
        for variant, col, lab in (("pure", ORANGE, T("只有 degradation + ramping", "degradation + ramping only")),
                                  ("minor", BLUE, T("保留当前最优的次要权重", "with the incumbent's minor weights"))):
            d = res[(b, variant)]
            a.scatter(GRID, d, s=9, color=col, alpha=0.35, linewidths=0, zorder=2)
            a.plot(GRID, roll(d, 11), color=col, lw=2, label=lab, zorder=3)
        r0 = to_ratio(inc[b])[0]
        a.axvline(r0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        a.text(r0 - 0.004, YMAX * 0.97, T(f"当前最优 r = {r0:.3f}", f"incumbent r = {r0:.3f}"), color=INK2, fontsize=8, va="top", ha="right")
        a.axhline(0, color=AXIS, lw=0.8, zorder=1)
        a.set_ylim(-0.4, YMAX); a.set_xlim(0.15, 0.50); a.set_title(bname[b])
        a.set_xlabel(T("r = degradation / (degradation + ramping)", "r = degradation / (degradation + ramping)"))
    ax[0].set_ylabel(T("R − 当前最优", "R − incumbent"))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=2, fontsize=9, handlelength=1.6, columnspacing=1.8)
    fig.suptitle(T("r 的细扫：点是单次评估，线是 11 点滑动中位数；纵轴截断在 8（r 再大时 1 号楼升到 +37.7）",
                   "Fine scan of r: dots are single evaluations, lines are 11-point running medians; y axis cut at 8"),
                 x=0.01, y=0.985, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig("mech_rscan.png", dpi=140); plt.close(fig)

    # --- dose response ---
    fig, ax = plt.subplots(3, 5, figsize=(14, 7.2), sharex=True, sharey=True)
    YMAX, X0 = 5.0, 4e-5
    for b in range(3):
        m0 = to_ratio(inc[b])[1]
        base = with_block(b, block(to_ratio(inc[b])[0], np.zeros(5)))
        for j in range(5):
            a = ax[b, j]
            g, d = add[(b, j)]
            a.plot(g, np.clip(d, None, YMAX * 2), color=ORANGE, lw=1.6, marker="o", ms=2.5, label=T("只加这一个次要权重", "this minor weight alone"))
            a.plot([X0], [base], marker="D", ms=5, color=ORANGE, clip_on=False, zorder=5)
            g, d = loo[(b, j)]
            a.plot(g[g > 0], np.clip(d[g > 0], None, YMAX * 2), color=BLUE, lw=1.6, marker="o", ms=2.5, label=T("其余次要权重保持当前值", "other minor weights as in the incumbent"))
            a.plot([X0], [d[0]], marker="D", ms=5, color=BLUE, clip_on=False, zorder=5)
            if m0[j] > 0:
                a.axvline(m0[j], color=INK2, lw=0.8, ls=(0, (3, 3)))
            a.axhline(0, color=AXIS, lw=0.8)
            a.set_xscale("log"); a.set_xlim(X0, 1.3); a.set_ylim(-0.3, YMAX)
            if b == 0:
                a.set_title(MN[j])
            if j == 0:
                a.set_ylabel(f"{bname[b]}\n" + T("R − 当前最优", "R − incumbent"))
            if b == 2:
                a.set_xlabel(T("权重 / (degradation + ramping)", "weight / (degradation + ramping)"))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2, fontsize=9, handlelength=1.6, columnspacing=1.8)
    fig.suptitle(T("次要权重的剂量曲线：菱形是该权重为 0 时的值，虚线是当前最优的取值；纵轴截断在 5",
                   "Dose curves of the minor weights: diamonds are the value at weight 0, dashed lines the incumbent's level; y axis cut at 5"),
                 x=0.01, y=0.99, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("mech_dose.png", dpi=140); plt.close(fig)

    # --- pair grids ---
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    labels = ["0", "1/8", "1/4", "1/2", "0.71", "1", "1.41", "2", "4", "8"]
    for b in range(3):
        a = ax[b]; j1, j2 = PAIRS[b]; M = pair[b]
        a.grid(False)
        im = a.imshow(np.clip(M, 0, 3), cmap=cmap, vmin=0, vmax=3, origin="lower", aspect="auto")
        for i in range(10):
            for k in range(10):
                if np.isfinite(M[i, k]):
                    a.text(k, i, f"{M[i, k]:.1f}" if abs(M[i, k]) >= 0.05 else "0", ha="center", va="center", fontsize=7.5,
                           color="#ffffff" if M[i, k] > 1.4 else INK)
        a.set_xticks(range(10)); a.set_xticklabels(labels); a.set_yticks(range(10)); a.set_yticklabels(labels)
        a.set_xlabel(T(f"{MN[j2]}（当前最优的倍数）", f"{MN[j2]} (multiple of the incumbent)"))
        a.set_ylabel(T(f"{MN[j1]}（当前最优的倍数）", f"{MN[j1]} (multiple of the incumbent)"))
        a.set_title(bname[b]); a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle(T("每栋楼最有价值的两个次要权重：格内数字是 R − 当前最优，颜色越深越差（3 以上同色）",
                   "The two most valuable minor weights per building: cells show R − incumbent, darker is worse (capped at 3)"),
                 x=0.01, y=0.99, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("mech_pair.png", dpi=140); plt.close(fig)
    print("\nwrote mech_rscan.png, mech_dose.png, mech_pair.png")


if __name__ == "__main__":
    res = rscan(); report_rscan(res)
    loo, add, pair = dose(); report_dose(loo, add, pair)
    allv = [v for d in res.values() for v in d] + [v for _, d in loo.values() for v in d] + [v for _, d in add.values() for v in d] + [v for M in pair.values() for v in M.ravel()]
    allv = np.array(allv, float)
    print(f"\n{int(np.isfinite(allv).sum())} of {len(allv)} design points evaluated; lowest delta {np.nanmin(allv):+.4f}; below incumbent: {int((allv < -1e-9).sum())}")
    if "--plot" in sys.argv:
        plots(res, loo, add, pair)
