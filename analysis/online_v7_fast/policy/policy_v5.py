"""15 轮加速策略：仅调整提案日程与名额分配，不读取任何历史运行结果。

相对 v4：
* 第 4 轮开始逐坐标扫描，提前利用阶梯边界与有效坐标。
* 每轮最多保留 12 个跨楼组合，并复用上一轮最多 6 个有效坐标移动。
* 第 1–8 轮保留模型探索下限；第 9–15 轮降低模型下限，把名额让给扫描、
  随机大步长和细部移动。具体名额仍由最近四轮的真实命中率决定。

数据边界与 v4 相同：初始只见 2048 条训练行，之后只见本次运行自己的结果。
"""
from __future__ import annotations

from dataclasses import dataclass

from policy_v4 import BlockDescentPolicyV4, CLASSES, ParamsV4


@dataclass
class ParamsV5(ParamsV4):
    scan_call: int = 4
    n_combo: int = 12
    n_icombo: int = 6
    late_model_call: int = 9
    model_floor: float = 0.25
    late_model_floor: float = 0.12
    random_floor: float = 0.12
    scan_floor: float = 0.20


class BlockDescentPolicyV5(BlockDescentPolicyV4):
    """v4 的加速日程版；移动生成、平台过滤和代理模型均沿用 v4。"""

    def __init__(self, params: ParamsV5 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV5(), seed)

    def _shares(self, hist, call_no):
        """按近期命中率分配自由名额，同时保留各搜索通道的最低探索量。"""
        p = self.p
        recent = hist[-p.share_window:]
        classes = [c for c in CLASSES if c != "scan" or call_no >= p.scan_call]
        rate = {}
        for c in classes:
            hits = sum(s[c][0] for s in recent if c in s)
            count = sum(s[c][1] for s in recent if c in s)
            rate[c] = (hits + 1) / (count + 5)  # 与 v4 相同的 Beta(1, 4) 先验
        total = sum(rate.values())
        raw = {c: rate[c] / total for c in classes}
        model_floor = p.model_floor if call_no < p.late_model_call else p.late_model_floor
        floors = {}
        for c in classes:
            if c == "model":
                floors[c] = model_floor
            elif c == "random":
                floors[c] = p.random_floor
            elif c == "scan":
                floors[c] = p.scan_floor
            else:
                floors[c] = p.share_floor
        free = 1.0 - sum(floors.values())
        if free < 0:
            raise ValueError(f"候选类别下限之和超过 1：{floors}")
        return {c: floors[c] + free * raw[c] for c in classes}
