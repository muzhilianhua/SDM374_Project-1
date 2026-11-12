"""Pre-submit check (T08): which proposed points may be sent in the next call."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..simplex import DIM, as_batch, check_weights
from .evaluator import renormalise

# Why a candidate was dropped, in the order the checks run.
REASONS = ("constraint", "training_row", "sent_before", "batch_duplicate")


@dataclass(frozen=True)
class Screening:
    accepted: tuple[int, ...]                    # candidate positions to send, in the policy's order
    rejected: tuple[dict, ...] = field(default_factory=tuple)   # {"index", "reason", "detail"}
    examined: int = 0                            # candidates looked at before n were accepted

    def counts(self) -> dict[str, int]:
        return {reason: sum(item["reason"] == reason for item in self.rejected) for reason in REASONS}


def _nearest_gap(point: np.ndarray, rows: np.ndarray) -> float:
    return float(np.abs(rows - point).max(axis=1).min()) if len(rows) else np.inf


def screen(candidates, *, train, sent, n: int, atol: float = 1e-9, accepted=None) -> Screening:
    """Keep the first `n` candidates that pass every check, in order.

    A point is dropped when it breaks the simplex constraint (`check_weights`), or when its renormalised
    weights are within `atol` (max abs difference) of a training row, of a point sent in an earlier call, or
    of a point already accepted for this batch. Renormalising first matters because the solver rescales each
    building, so W and 2·W are one point (decision 20 forbids re-evaluating a point). `accepted` holds
    points already taken for this batch in an earlier screening round; they count as batch duplicates.
    """
    candidates = as_batch(candidates) if len(np.asarray(candidates)) else np.empty((0, DIM))
    train = renormalise(as_batch(train))
    sent = renormalise(as_batch(sent)) if len(sent) else np.empty((0, DIM))
    kept = (renormalise(as_batch(accepted)) if accepted is not None and len(accepted)
            else np.empty((0, DIM)))
    accepted: list[int] = []
    rejected: list[dict] = []
    examined = 0
    for index, row in enumerate(candidates):
        if len(accepted) == n:
            break
        examined += 1
        try:
            check_weights(row)
        except ValueError as error:
            rejected.append({"index": index, "reason": "constraint", "detail": str(error)})
            continue
        point = renormalise(row.reshape(1, DIM))[0]
        for reason, rows in (("training_row", train), ("sent_before", sent), ("batch_duplicate", kept)):
            gap = _nearest_gap(point, rows)
            if gap <= atol:
                rejected.append({"index": index, "reason": reason, "detail": f"gap {gap:.3g}"})
                break
        else:
            accepted.append(index)
            kept = np.vstack([kept, point])
    return Screening(tuple(accepted), tuple(rejected), examined)
