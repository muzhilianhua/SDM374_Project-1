"""v7：保留 v4 的完整批次，仅把零产出的细步/零权重/冗余组合名额换成多尺度候选。

复盘依据（v4 第 12–20 轮真实日志）：
* 新低点来自 model(1)、scan(4)、icombo(3)、random(2)、tiny(1)、base(3)；
* fine、zero、combo 在 9 轮内没有产生任何新低；
* v6 的问题正是把 model 池改成含 block 的混合池、并压缩了 random 名额，
  导致第 12 轮丢掉了 model 给出的 0.99 大跳。

因此 v7 的执行方式是：
1. 第 1–11 轮与 v4 完全一致（冻结，找盆地）；
2. 第 12–20 轮先调用 v4 的 propose 得到完整 50 点（含全部 model/scan/random/icombo）；
3. 按 fine → zero → combo 的优先级腾出名额，填入 coarse / block / restart 三类多尺度候选；
   每栋楼优先保证 1 个 coarse、1 个 block、1 个 restart，再按名额补充第二个 coarse。

任何替换都经过去重校验：若新候选与训练集、已测点或保留点重复，则放弃该替换并保留 v4 原点。
所有随机性仍只由 (seed, call_no) 决定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import MINOR, ZERO, bkey, norm_blocks
from policy_v4 import BlockDescentPolicyV4, ParamsV4
from policy_v6 import BlockDescentPolicyV6, ParamsV6, COARSE_F

# 替换优先级：先动 fine，再动 zero，最后动 combo（base 永不动）
REPLACE_ORDER = ("fine", "zero")
COARSE_STEPS = [0.45, 0.6, 0.75, 1.35, 1.7, 2.2, 3.0]


@dataclass
class ParamsV7(ParamsV6):
    freeze_calls: int = 11
    base_sigma: float = 0.22
    wide_sigma: float = 0.35
    max_replace: int = 12


class BlockDescentPolicyV7(BlockDescentPolicyV6):
    def __init__(self, params: ParamsV7 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV7(), seed)
        self._v4 = BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3), seed=seed)

    # ------------------------------------------------------------------ 入口
    def propose(self, st, n: int = 50):
        if len(st.calls) + 1 <= self.p.freeze_calls:
            return self._v4.propose(st, n)
        return self._propose_hybrid(st, n)

    # ------------------------------------------------------------------ 多尺度候选
    def _coarse_block(self, rng, base_blk, k):
        w = base_blk.copy()
        w[k] *= COARSE_STEPS[rng.integers(len(COARSE_STEPS))]
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        w[w < ZERO] = 0.0
        return w / w.sum() if w.sum() > 0 else None

    def _restart_block(self, rng, base_blk, seeds):
        return self._blend_block(rng, base_blk, seeds)

    # ------------------------------------------------------------------ 主流程
    def _propose_hybrid(self, st, n):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 7])
        inc, _, val, blocks, scale = self._replay(st)
        stagnant = self._stagnant(st)
        train_top = self._top_train_blocks(st)

        # 1) v4 的完整批次：model/scan/random/icombo 全部保留
        #    v4.propose 返回 (props, info)，props 为 [(W 3x7, meta), ...]
        props, info = self._v4.propose(st, n)
        props = [(np.asarray(W, float), dict(m)) for W, m in props]
        metas = [m for _, m in props]

        # 2) 可替换名额：fine → zero → combo（base 与其余通道不动）
        replaceable = []
        for move in REPLACE_ORDER:
            replaceable += [i for i, m in enumerate(metas)
                            if m.get("kind") == "single" and m.get("move") == move]
        replaceable += [i for i, m in enumerate(metas) if m.get("kind") == "combo"]
        replaceable = replaceable[: p.max_replace]
        if not replaceable:
            info.update({"phase": "v4_batch", "replaced": 0})
            return props[:n], info

        # 3) 生成多尺度候选，轮转覆盖三栋楼
        W = np.array([norm_blocks(W)[0] for W, _ in props])
        seen = {tuple(bkey(w[b]) for b in range(3)) for w in W}
        seen |= {tuple(bkey(w[b]) for b in range(3)) for w in norm_blocks(st.X_train)}
        for c in st.calls:
            for e in c:
                seen.add(tuple(bkey(e["W"][b]) for b in range(3)))

        def seeds_for(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items] + [train_top[b][i] for i in range(len(train_top[b]))]

        plans = [(0, "coarse1"), (1, "coarse1"), (2, "coarse1"),
                 (0, "block"), (1, "block"), (2, "block"),
                 (0, "restart"), (1, "restart"), (2, "restart"),
                 (0, "coarse2"), (1, "coarse2"), (2, "coarse2")]
        counts = {"coarse": 0, "block": 0, "restart": 0}
        used, new_rows = set(), []
        for b, kind in plans:
            if len(new_rows) >= len(replaceable):
                break
            sigma = p.wide_sigma if stagnant[b] else p.base_sigma
            if kind.startswith("coarse"):
                pos = np.flatnonzero(inc[b] > 0)
                if len(pos) == 0:
                    continue
                k = int(pos[rng.integers(len(pos))]) if kind == "coarse1" else int(
                    pos[np.argmax(inc[b][pos])])
                blk = self._coarse_block(rng, inc[b], k)
                move = "coarse"
            elif kind == "block":
                blk = self._move_block(rng, inc[b], sigma)
                move = "block"
            else:
                blk = self._restart_block(rng, inc[b], seeds_for(b))
                move = "restart"
            if blk is None or np.allclose(blk, inc[b], atol=1e-12):
                continue
            Wnew = inc.copy()
            Wnew[b] = blk
            key = tuple(bkey(Wnew[j]) for j in range(3))
            if key in seen or key in used:
                continue
            used.add(key)
            new_rows.append((Wnew, {"kind": "single", "building": b, "move": move}))
            counts[move] += 1

        # 4) 按优先级替换，未填满的名额保留 v4 原点
        replaced = 0
        for idx, row in zip(replaceable, new_rows):
            props[idx] = row
            metas[idx] = row[1]
            replaced += 1

        info.update({"phase": "v4_batch_plus_multiscale", "replaced": replaced,
                     "coarse": counts["coarse"], "block": counts["block"],
                     "restart": counts["restart"], "stagnant": stagnant,
                     "sigma": [round(p.wide_sigma if s else p.base_sigma, 3) for s in stagnant]})
        return props[:n], info
