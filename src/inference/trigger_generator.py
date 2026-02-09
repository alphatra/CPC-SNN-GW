from __future__ import annotations

from typing import Dict, Iterable, List


def generate_triggers(
    predictions: Iterable[Dict[str, float | str]],
    threshold: float = 0.5,
    min_gap: int = 0,
) -> List[Dict[str, float | str]]:
    """
    Converts per-window probabilities into trigger list.

    Args:
        predictions: iterable of dicts with keys {"id", "prob_signal"}
        threshold: minimum probability to emit trigger
        min_gap: suppress triggers closer than this many windows (by index order)
    """
    triggers: List[Dict[str, float | str]] = []
    last_kept_idx = -10**9

    for idx, rec in enumerate(predictions):
        p = float(rec["prob_signal"])
        if p < threshold:
            continue
        if idx - last_kept_idx <= min_gap:
            continue
        triggers.append(
            {
                "id": rec["id"],
                "prob_signal": p,
                "rank": len(triggers) + 1,
            }
        )
        last_kept_idx = idx
    return triggers
