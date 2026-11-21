"""Build the report figures from committed real-evaluator logs (no solver calls).

Inputs:
- `analysis/online_rehearsal/rehearsals/<variant>_s<seed>/online_log.jsonl` (T37, PR #25)
- the submitted policy's log, `analysis/online_v7/runs/explore_v7_s7/online_log.jsonl` (PR #29 / #32);
  pass `--h7-log` while that folder is not on this branch.
- h7 on seeds 2026 and 11, `analysis/online_v7_seeds/h7_s<seed>/checkpoints/call_<kk>/result.json`

Usage (repository root):
    uv run python report/figures/make_figures.py [--h7-log PATH] [--out report/figures]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REHEARSALS = ROOT / "analysis" / "online_rehearsal" / "rehearsals"
H7_DEFAULT = ROOT / "analysis" / "online_v7" / "runs" / "explore_v7_s7" / "online_log.jsonl"
H7_SEEDS = ROOT / "analysis" / "online_v7_seeds"
BEST_TRAIN = 115.76481925917324   # p01969, the offline bar (decision 17)
OFFLINE_BEST = 115.68             # offline submission 2
COLUMN = (3.45, 2.25)             # IEEE single column, inches


def read_points(path: Path) -> dict[int, list[tuple[float, str]]]:
    """call -> [(R, move)] for every evaluated point of one run."""
    calls: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "point":
            calls[int(record["call"])].append((float(record["R"]), str(record.get("meta", {}).get("move", ""))))
    return dict(calls)


def read_checkpoints(folder: Path) -> dict[int, list[tuple[float, str]]]:
    """Same as read_points, for a run of the graded package (one result.json per call)."""
    calls: dict[int, list[tuple[float, str]]] = {}
    for result in sorted(folder.glob("call_*/result.json")):
        record = json.loads(result.read_text(encoding="utf-8"))
        calls[int(record["call"])] = [(float(r), "") for r in record["R"]]
    return calls


def best_so_far(calls: dict[int, list[tuple[float, str]]]) -> np.ndarray:
    minima = [min(r for r, _ in calls[c]) for c in sorted(calls)]
    return np.minimum.accumulate(minima)


def convergence(h7: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=COLUMN)
    styles = {"v1": ("#9aa5b1", ":"), "v2": ("#6b8fb3", "--"), "v4": ("#2f6f9f", "-")}
    for variant, (color, dash) in styles.items():
        for index, run in enumerate(sorted(REHEARSALS.glob(f"{variant}_s*"))):
            curve = best_so_far(read_points(run / "online_log.jsonl"))
            ax.plot(np.arange(1, len(curve) + 1), curve, color=color, ls=dash, lw=0.9, alpha=0.9,
                    label=f"{variant} (3 seeds)" if index == 0 else None)
    for index, run in enumerate(sorted(H7_SEEDS.glob("h7_s*"))):
        curve = best_so_far(read_checkpoints(run / "checkpoints"))
        ax.plot(np.arange(1, len(curve) + 1), curve, color="#c0392b", lw=0.9, alpha=0.8,
                label="h7, seeds 2026 and 11" if index == 0 else None)
    curve = best_so_far(h7)
    ax.plot(np.arange(1, len(curve) + 1), curve, color="#c0392b", lw=1.8, label="submitted h7, seed 7")
    ax.axhline(OFFLINE_BEST, color="k", lw=0.8, ls="-.")
    ax.text(20, OFFLINE_BEST + 0.25, "best offline 115.68", ha="right", va="bottom", fontsize=7)
    ax.axvline(11.5, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.text(11.7, 102.4, "h7 hybrid phase →", fontsize=6.5, color="#c0392b")
    ax.set_xlabel("evaluator call")
    ax.set_ylabel("best R so far (kWh)")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(102, 117)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)


def batches(h7: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=COLUMN)
    rng = np.random.default_rng(0)
    for call in sorted(h7):
        values = np.array([r for r, _ in h7[call]])
        ax.scatter(call + rng.uniform(-0.25, 0.25, len(values)), np.minimum(values, 139.5), s=2.5,
                   color="#9aa5b1", alpha=0.6, lw=0)
    curve = best_so_far(h7)
    ax.step(np.arange(1, len(curve) + 1), curve, where="mid", color="#c0392b", lw=1.4, label="batch minimum (best so far)")
    medians = [np.median([r for r, _ in h7[c]]) for c in sorted(h7)]
    ax.plot(sorted(h7), medians, color="#2f6f9f", lw=1.0, ls="--", label="batch median")
    ax.axhline(BEST_TRAIN, color="k", lw=0.8, ls="-.")
    ax.text(20.4, BEST_TRAIN + 0.6, "best training row", ha="right", fontsize=6.5)
    ax.set_xlabel("evaluator call (50 points each)")
    ax.set_ylabel("measured R (kWh)")
    ax.set_xlim(0.4, 20.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(101, 140)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "h7_batches.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h7-log", type=Path, default=H7_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plt.style.use(str(ROOT / "report" / "style.mplstyle"))
    h7 = read_points(args.h7_log)
    if sorted(h7) != list(range(1, 21)) or any(len(v) != 50 for v in h7.values()):
        raise SystemExit("the h7 log must hold 20 calls of 50 points")
    args.out.mkdir(parents=True, exist_ok=True)
    convergence(h7, args.out)
    batches(h7, args.out)
    print(f"h7 best R = {best_so_far(h7)[-1]:.10f}; figures in {args.out}")


if __name__ == "__main__":
    main()
