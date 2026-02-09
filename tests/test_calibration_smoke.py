import numpy as np

from src.evaluation.calibration import (
    CalibrationArtifact,
    apply_calibration_to_logits,
    fit_isotonic_calibrator,
    fit_temperature_scaler,
)


def test_temperature_scaler_smoke():
    logits = np.array([-3.0, -1.0, -0.5, 0.1, 0.8, 1.5, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    t = fit_temperature_scaler(logits, labels, max_iter=50, lr=0.05)
    assert np.isfinite(t)
    assert t > 0.0


def test_isotonic_and_apply_smoke():
    logits = np.array([-2.0, -1.1, -0.3, 0.2, 0.6, 1.1, 2.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    x, y = fit_isotonic_calibrator(probs, labels)
    artifact = CalibrationArtifact(method="isotonic", temperature=1.0, isotonic_x=x, isotonic_y=y)
    p_cal = apply_calibration_to_logits(logits, artifact)
    assert np.all(np.isfinite(p_cal))
    assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))
