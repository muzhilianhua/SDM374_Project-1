"""Build the report figures from committed real-simulator rehearsal logs (no solver calls).

Input: `analysis/online_rehearsal/rehearsals/<variant>_s<seed>/online_log.jsonl` (T37 and the 2026-11-22 runs);
every seed present for a variant is used. The statistics come from `report/evidence.py`.

Usage (repository root):
    uv run python report/figures/make_figures.py [--out report/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "report"))
from evidence import REH, analyse  # noqa: E402

OFFLINE_BEST = 115.68             # offline submission 2
COLUMN = (3.45, 2.2)              # IEEE single column, inches
STYLE = {"v1": ("#9aa5b1", ":", "v1: block descent"),
         "v2": ("#6b8fb3", "--", "v2: + training shift bias"),
         "v4": ("#c0392b", "-", "v4: + evidence-driven slots"),
         "v4j": ("#2f6f9f", "-.", "v4j: v4 without blocks")}


def runs_of(variant: str) -> list[dict]:
    out = []
    for run in sorted(REH.glob(f"{variant}_s*")):
        if run.name.rsplit("_s", 1)[0] == variant and (run / "online_log.jsonl").exists():
            r = analyse(run.name)
            if len(r["curve"]) == 20:
                out.append(r)
    return out


def convergence(out: Path) -> None:
    fig, ax = plt.subplots(figsize=COLUMN)
    x = np.arange(1, 21)
    for variant, (color, dash, label) in STYLE.items():
        rs = runs_of(variant)
        if not rs:
            continue
        c = np.array([r["curve"] for r in rs])
        ax.plot(x, c.mean(axis=0), color=color, ls=dash, lw=1.4 if variant in ("v4", "v4j") else 1.0,
                label=f"{label} (n={len(rs)})")
        if variant in ("v4", "v4j", "v2"):
            ax.fill_between(x, c.min(axis=0), c.max(axis=0), color=color, alpha=0.12, lw=0)
    ax.axhline(OFFLINE_BEST, color="k", lw=0.8, ls=(0, (1, 2)))
    ax.text(20, OFFLINE_BEST + 0.2, "best offline 115.68", ha="right", va="bottom", fontsize=6.5)
    ax.set_xlabel("evaluator call (50 points each)")
    ax.set_ylabel("best R so far (kWh)")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(102.5, 116.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=6.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)


def allocation(out: Path) -> None:
    classes = ["model", "random", "fine", "scan", "nav", "combination"]
    names = ["surr.", "random", "fine", "scan", "shift", "combo"]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    ax2 = ax.twinx()
    width = 0.38
    x = np.arange(len(classes))
    for i, variant in enumerate(("v2", "v4")):
        hits = {}
        for r in runs_of(variant):
            for k, (h, n) in r["hits"].items():
                a = hits.setdefault(k, [0, 0]); a[0] += h; a[1] += n
        total = sum(n for _, n in hits.values())
        share = [hits.get(k, [0, 0])[1] / total for k in classes]
        rate = [hits[k][0] / hits[k][1] if hits.get(k, [0, 0])[1] else np.nan for k in classes]
        color = STYLE[variant][0]
        ax.bar(x + (i - 0.5) * width, np.array(share) * 100, width, color=color, alpha=0.75,
               label=f"{variant} slot share")
        ax2.plot(x + (i - 0.5) * width, np.array(rate) * 100, ls="none", marker="o" if variant == "v2" else "D",
                 ms=3.5, color="k" if variant == "v4" else "#555555", mfc="white" if variant == "v2" else "k",
                 label=f"{variant} hit rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("share of points (%)")
    ax2.set_ylabel("hit rate (%), log scale")
    ax2.set_yscale("log")
    ax2.set_ylim(1, 100)
    ax.set_ylim(0, 50)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              fontsize=5.6, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "allocation.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plt.style.use(str(ROOT / "report" / "style.mplstyle"))
    args.out.mkdir(parents=True, exist_ok=True)
    convergence(args.out)
    allocation(args.out)
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
