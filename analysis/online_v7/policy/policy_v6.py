"""v6：前 11 轮完全沿用 v4，第 12 轮起在盆地内使用多尺度步长收敛。

设计依据（来自 35 轮真实日志的复盘）：
* v4 的 1–11 轮完成了从 115.76 到 105.11 的盆地定位，第 12 轮出现 0.99 的大跳；
* 进入盆地后若只做单坐标微调，会停在阶梯平台上：v5 在 107.13 处的 311 个近邻无一更低；
* 因此盆地内需要同时保留「大步长整块移动」和「更细的坐标精度」，而不是只用其一。

三类通道：
1. 粗跨步 (coarse)：单坐标乘以 0.2–3.0 的粗网格步长，用于跨过阶梯平台；
2. 块级重整 (block/restart)：同时改 2–3 个坐标，或把当前块与训练集低 R 块、本轮精英块做凸组合，
   用于整块换结构；
3. 细步 (fine)：0.4%–6% 的坐标微调与极小权重复活，只在已确认的空步区间之外取样，用于抓窄凹口。

步长自适应：某栋楼连续 `stagnation_window` 轮没有单楼改进时，该楼的粗步长与块步长放大到
`wide_sigma`；一旦出现改进则回落到 `base_sigma`。所有随机性只由 (seed, call_no) 决定。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from policy import MINOR, ZERO, TrainPriors, bkey, norm_blocks
from policy_v4 import BlockDescentPolicyV4, ParamsV4, one_coord
from surrogate import Surrogate

COARSE_F = [0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.15, 1.35, 1.7, 2.2, 3.0]
FINE_F = [0.94, 0.96, 0.975, 0.985, 0.992, 0.996, 1.004, 1.008, 1.015, 1.025, 1.04, 1.06]


@dataclass
class ParamsV6(ParamsV4):
    freeze_calls: int = 11        # 前 11 轮与 v4 逐点相同
    stagnation_window: int = 3    # 连续多少个无改进轮次视为停滞
    base_sigma: float = 0.18      # 正常块移动的 log10 步长
    wide_sigma: float = 0.32      # 停滞时的块移动步长（约 1 倍数量级范围）
    share_model: float = 0.38
    share_coarse: float = 0.22
    share_fine: float = 0.22
    n_combo: int = 6
    pool: int = 1500


class BlockDescentPolicyV6(BlockDescentPolicyV4):
    def __init__(self, params: ParamsV6 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV6(), seed)
        # 冻结阶段直接委托给 v4 实例，保证第 1–11 轮与其逐点一致。
        self._frozen = BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3), seed=seed)
        self._train_top_blocks = None

    # ------------------------------------------------------------------ 入口
    def propose(self, st, n: int = 50):
        if len(st.calls) + 1 <= self.p.freeze_calls:
            return self._frozen.propose(st, n)
        return self._propose_in_basin(st, n)

    # ------------------------------------------------------------------ 停滞判定
    def _stagnant(self, st):
        """每栋楼在最近 window 轮内是否没有任何单楼改进。"""
        window = self.p.stagnation_window
        hit = [False] * 3
        seen = 0
        for call in reversed(st.calls):
            if seen >= window:
                break
            base = next((e["R"] for e in call if e["meta"].get("kind") == "base"), None)
            if base is None:
                continue
            for e in call:
                m = e["meta"]
                if m.get("kind") == "single" and np.isfinite(e["R"]) and e["R"] < base - 1e-9:
                    hit[m["building"]] = True
            seen += 1
        return [not h for h in hit]

    # ------------------------------------------------------------------ 训练集低 R 块（只用 2048 行）
    def _top_train_blocks(self, st):
        if self._train_top_blocks is not None:
            return self._train_top_blocks
        W = norm_blocks(st.X_train)
        order = np.argsort(st.y_train)[:10]
        self._train_top_blocks = [W[order][:, b, :].copy() for b in range(3)]
        return self._train_top_blocks

    # ------------------------------------------------------------------ 新移动
    def _move_block(self, rng, base_blk, sigma):
        """同时改动 2–3 个坐标的大步长移动，可选把一个次要目标清零。"""
        w = base_blk.copy()
        pos = np.flatnonzero(w > 0)
        if len(pos) == 0:
            return None
        nch = 2 if rng.random() < 0.7 else 3
        ks = rng.choice(pos, size=min(nch, len(pos)), replace=False)
        factors = np.clip(10 ** rng.normal(0, sigma, len(ks)), 0.2, 4.0)
        w[ks] *= factors
        if rng.random() < 0.30 and len(pos) > 2:
            minor = [k for k in pos if k in MINOR]
            if minor:
                w[rng.choice(minor)] = 0.0
        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            return None
        w = w / w.sum()
        w[w < ZERO] = 0.0
        return w / w.sum() if w.sum() > 0 else None

    def _blend_block(self, rng, base_blk, seeds):
        """与训练集/本轮精英块做凸组合，用于整块换结构。"""
        if not seeds:
            return None
        a = float(rng.choice([0.3, 0.5, 0.7]))
        src = seeds[rng.integers(len(seeds))]
        w = (1 - a) * base_blk + a * src
        w[w < ZERO] = 0.0
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return w

    # ------------------------------------------------------------------ 盆地内提案
    def _propose_in_basin(self, st, n):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 6])
        priors = TrainPriors.from_training(st.X_train, st.y_train if p.shift_bias else None)
        inc, R_inc, val, blocks, scale = self._replay(st)
        null, _, improvers = self._learn(st)
        stagnant = self._stagnant(st)
        train_top = self._top_train_blocks(st)

        Xo = [e["W"].reshape(21) for c in st.calls for e in c if np.isfinite(e["R"])]
        yo = [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])]
        X = np.vstack([st.X_train] + ([np.array(Xo)] if Xo else []))
        y = np.concatenate([st.y_train, np.array(yo)]) if yo else st.y_train.copy()
        model = Surrogate(local_rows=p.local_rows, seed=self.seed).fit(X, y, center=inc.reshape(21))

        out, seen = [], set()
        sent = {tuple(bkey(e["W"][b]) for b in range(3)) for c in st.calls for e in c}
        seen |= sent
        seen.add(tuple(bkey(inc[b]) for b in range(3)))
        Wt = norm_blocks(st.X_train)
        seen |= {tuple(bkey(w[b]) for b in range(3)) for w in Wt}

        def add(W, meta):
            W = norm_blocks(W)[0]
            k = tuple(bkey(W[b]) for b in range(3))
            if k in seen:
                return False
            seen.add(k)
            out.append((W, meta))
            return True

        def in_null(b, base_blk, blk):
            oc = one_coord(base_blk, blk)
            if oc is None:
                return False
            k, kind, v = oc
            nl = null.get((b, bkey(base_blk), k))
            if nl is None:
                return False
            return (nl["lo"] <= v <= nl["hi"]) if kind == "f" else (v <= nl["tmax"])

        def elites(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items]

        # ---- base 与跨楼组合（与 v4 相同）
        v_inc = [val[b][bkey(inc[b])] for b in range(3)]
        better = []
        for b in range(3):
            items = sorted(val[b].items(), key=lambda kv: kv[1])
            better.append([(blocks[b][k], v - v_inc[b]) for k, v in items if v < v_inc[b] - 1e-9][:2])
        base = inc.copy()
        combos = []
        for choice in itertools.product(*[[(inc[b], 0.0)] + better[b] for b in range(3)]):
            W = np.array([c[0] for c in choice])
            if all(bkey(W[b]) == bkey(inc[b]) for b in range(3)):
                continue
            combos.append((R_inc + sum(c[1] for c in choice), W))
        combos.sort(key=lambda t: t[0])
        combos = [(pr, W) for pr, W in combos if tuple(bkey(W[b]) for b in range(3)) not in sent]
        if combos:
            base = combos[0][1]
            add(base, {"kind": "base", "pred": combos[0][0]})
            for pr, W in combos[1: 1 + p.n_combo]:
                add(W, {"kind": "combo", "pred": pr})

        # ---- 代理池：移动类型含大跨步与块移动
        pools, sigma_use = {}, {}
        for b in range(3):
            sigma_use[b] = p.wide_sigma if stagnant[b] else p.base_sigma
            parents = [base[b]] + [e for e in elites(b) if bkey(e) != bkey(base[b])]
            cand = []
            kinds = ["scale1", "scale1", "multi", "shift", "block", "block", "tiny"]
            for _ in range(p.pool):
                par = parents[0] if rng.random() < 0.7 or len(parents) == 1 else parents[rng.integers(1, len(parents))]
                kind = kinds[rng.integers(len(kinds))]
                if kind == "block":
                    blk = self._move_block(rng, par, sigma_use[b])
                else:
                    blk = self._move(rng, par, max(scale[b], 0.35), kind, priors, b=b)
                if blk is not None:
                    cand.append(blk)
            if not cand:
                continue
            cand = np.array(cand)
            Wfull = np.repeat(base[None], len(cand), 0)
            Wfull[:, b] = cand
            mu, sd = model.predict(Wfull.reshape(-1, 21))
            pools[b] = (cand, mu, sd)

        # ---- 名额分配：模型 / 粗跨步 / 细步 / 块移动
        slots = n - len(out)
        per = [slots // 3 + (1 if b < slots % 3 else 0) for b in range(3)]
        for b in range(3):
            if b not in pools:
                continue
            k_model = max(2, int(round(per[b] * p.share_model)))
            k_coarse = max(2, int(round(per[b] * p.share_coarse)))
            k_fine = max(2, int(round(per[b] * p.share_fine)))
            k_block = max(1, per[b] - k_model - k_coarse - k_fine)
            self._fill_model(b, base, pools[b], k_model, add)
            self._fill_coarse(b, base, k_coarse, rng, call_no, sigma_use[b], in_null, add)
            self._fill_fine(b, base, k_fine, null, priors, rng, call_no, in_null, add)
            self._fill_blocks(b, base, k_block, rng, call_no, sigma_use[b],
                              train_top[b], elites(b), add)

        # ---- 补齐
        tries = 0
        while len(out) < n and tries < 3000:
            tries += 1
            b = int(rng.integers(3))
            blk = self._move_block(rng, base[b], p.base_sigma)
            if blk is None or in_null(b, base[b], blk):
                continue
            W = base.copy()
            W[b] = blk
            add(W, {"kind": "single", "building": b, "move": "fill"})

        info = {"call": call_no, "R_inc": R_inc, "scale": [round(s, 4) for s in scale],
                "model_secs": model.fit_secs, "stagnant": stagnant,
                "sigma": [round(sigma_use[b], 3) for b in range(3)],
                "phase": "in_basin_multiscale"}
        return out[:n], info

    # ------------------------------------------------------------------ 三类填充
    def _fill_coarse(self, b, base, k, rng, call_no, sigma, in_null, add):
        """单坐标粗跨步：故意越过已确认的空步区间。"""
        blk0 = base[b]
        got, depth = 0, 0
        order = [(call_no * 3 + b + j) % 7 for j in range(7)]
        while got < k and depth < 4:
            for kk in order:
                if got >= k:
                    break
                if blk0[kk] <= 0:
                    continue
                idx = (depth * 3 + kk) % len(COARSE_F)
                w = blk0.copy()
                w[kk] *= COARSE_F[idx]
                w = w / w.sum()
                if in_null(b, blk0, w):
                    continue
                W = base.copy()
                W[b] = w
                got += add(W, {"kind": "single", "building": b, "move": "coarse"})
            depth += 1

    def _fill_fine(self, b, base, k, null, priors, rng, call_no, in_null, add):
        """细步：0.4%–6% 坐标微调 + 极小权重复活，跳过已知空步区间。"""
        blk0 = base[b]
        got, depth = 0, 0
        order = [(call_no + b + j) % 7 for j in range(7)]
        tiny_grid = np.geomspace(priors.tiny_lo, max(priors.tiny_hi, 3 * priors.tiny_lo), 14)
        while got < k and depth < 10:
            for kk in order:
                if got >= k:
                    break
                nl = null.get((b, bkey(blk0), kk), {"lo": 1.0, "hi": 1.0, "tmax": 0.0})
                w = blk0.copy()
                if w[kk] > 0:
                    ups = [f for f in FINE_F if f > nl["hi"]]
                    downs = [f for f in sorted(FINE_F, reverse=True) if f < nl["lo"]]
                    steps = [v for pair in itertools.zip_longest(ups, downs) for v in pair if v is not None]
                    if depth >= len(steps):
                        continue
                    w[kk] *= steps[depth]
                else:
                    ts = [t for t in tiny_grid if t > nl["tmax"]]
                    if depth >= len(ts):
                        continue
                    w[kk] = ts[depth] * (w[4] + w[5])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                w[w < ZERO] = 0.0
                if w.sum() <= 0 or in_null(b, blk0, w):
                    continue
                W = base.copy()
                W[b] = w / w.sum()
                got += add(W, {"kind": "single", "building": b, "move": "fine"})
            depth += 1

    def _fill_blocks(self, b, base, k, rng, call_no, sigma, train_blocks, elite_blocks, add):
        """块级移动：2–3 坐标同步缩放；最后一个名额用于与训练/精英块做凸组合。"""
        got, tries = 0, 0
        seeds = list(elite_blocks) + [train_blocks[i] for i in range(len(train_blocks))]
        while got < k and tries < 200:
            tries += 1
            if got == k - 1 and seeds:
                w = self._blend_block(rng, base[b], seeds)
                move = "restart"
            else:
                w = self._move_block(rng, base[b], sigma)
                move = "block"
            if w is None:
                continue
            W = base.copy()
            W[b] = w
            got += add(W, {"kind": "single", "building": b, "move": move})
