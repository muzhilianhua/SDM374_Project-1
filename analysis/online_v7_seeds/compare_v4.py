"""Compare the h7 runs here with the T37 v4 rehearsals call by call (best R, and whether each batch is identical).

Usage: python analysis/online_v7_seeds/compare_v4.py [path/to/online_rehearsal/rehearsals]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "online_rehearsal" / "rehearsals"
for seed in (2026, 11):
    v4 = {}
    for line in open(REH / f"v4_s{seed}" / "online_log.jsonl"):
        r = json.loads(line)
        if r.get("type") == "point": v4.setdefault(r["call"], []).append((r["W"], r["R"]))
    ck = HERE / f"h7_s{seed}" / "checkpoints"
    h7 = {}
    for c in range(1, 21):
        res = json.load(open(ck / f"call_{c:02d}" / "result.json"))
        h7[c] = list(zip(res["weights"], res["R"]))
    best_h, best_v = float("inf"), float("inf"); rows = []
    first_diff = None
    for c in range(1, 21):
        bh = min(r for _, r in h7[c]); best_h = min(best_h, bh)
        bv = min(r for _, r in v4.get(c, [(None, float("inf"))])); best_v = min(best_v, bv)
        sameW = [w for w, _ in h7[c]] == [w for w, _ in v4.get(c, [])]
        sameR = [r for _, r in h7[c]] == [r for _, r in v4.get(c, [])]
        if not (sameW and sameR) and first_diff is None: first_diff = c
        rows.append((c, bh, best_h, bv, best_v, sameW, sameR))
    print(f"seed {seed}: h7 best {best_h:.6f}  v4 best {best_v:.6f}  diff {best_h-best_v:+.6f}  first differing call {first_diff}")
    for c, bh, cbh, bv, cbv, sw, sr in rows:
        print(f"  call {c:2d}  h7 min {bh:.4f} best {cbh:.4f} | v4 min {bv:.4f} best {cbv:.4f} | sameW={sw} sameR={sr}")
