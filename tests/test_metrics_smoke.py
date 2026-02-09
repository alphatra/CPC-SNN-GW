import numpy as np

from src.evaluation.metrics import compute_brier_score, compute_ece, compute_tpr_at_fpr


def test_metrics_api_smoke():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probs = np.array([0.01, 0.02, 0.1, 0.7, 0.8, 0.9], dtype=np.float64)

    tpr = compute_tpr_at_fpr(labels, probs, fpr_thresholds=[1e-3, 1e-2, 1e-1])
    assert 1e-3 in tpr and 1e-1 in tpr
    assert 0.0 <= tpr[1e-1]["tpr"] <= 1.0

    ece = compute_ece(labels, probs, n_bins=5)
    brier = compute_brier_score(labels, probs)
    assert ece >= 0.0
    assert brier >= 0.0
