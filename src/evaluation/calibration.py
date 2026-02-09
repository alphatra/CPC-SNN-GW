from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationArtifact:
    method: str
    temperature: float = 1.0
    isotonic_x: Optional[np.ndarray] = None
    isotonic_y: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        payload = {
            "version": 1,
            "method": self.method,
            "temperature": float(self.temperature),
        }
        if self.isotonic_x is not None and self.isotonic_y is not None:
            payload["isotonic_x"] = [float(v) for v in self.isotonic_x.tolist()]
            payload["isotonic_y"] = [float(v) for v in self.isotonic_y.tolist()]
        return payload

    @staticmethod
    def from_dict(payload: Dict) -> "CalibrationArtifact":
        ix = payload.get("isotonic_x")
        iy = payload.get("isotonic_y")
        return CalibrationArtifact(
            method=str(payload.get("method", "temperature")),
            temperature=float(payload.get("temperature", 1.0)),
            isotonic_x=None if ix is None else np.asarray(ix, dtype=np.float64),
            isotonic_y=None if iy is None else np.asarray(iy, dtype=np.float64),
        )


def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Learns scalar temperature T>0 minimizing BCE(sigmoid(logits/T), labels).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_temp) + 1e-6
        scaled_logits = logits_t / temperature
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels_t)
        loss.backward()
        optimizer.step()

    temperature = float((torch.exp(log_temp) + 1e-6).detach().cpu().item())
    return max(temperature, 1e-4)


def fit_isotonic_calibrator(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits isotonic calibration mapping p -> p_cal and returns thresholds.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(probs.astype(np.float64), labels.astype(np.float64))
    x = np.asarray(iso.X_thresholds_, dtype=np.float64)
    y = np.asarray(iso.y_thresholds_, dtype=np.float64)
    return x, y


def apply_isotonic_from_thresholds(probs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return probs
    if len(x) == 1:
        return np.full_like(probs, fill_value=float(y[0]), dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))


def apply_calibration_to_logits(logits: np.ndarray, artifact: CalibrationArtifact) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    t = max(float(artifact.temperature), 1e-6)
    probs = 1.0 / (1.0 + np.exp(-(logits / t)))
    if artifact.method in ("isotonic", "temperature_isotonic"):
        probs = apply_isotonic_from_thresholds(probs, artifact.isotonic_x, artifact.isotonic_y)
    return np.clip(probs, 1e-7, 1.0 - 1e-7)


def apply_calibration_to_probs(probs: np.ndarray, artifact: CalibrationArtifact) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if artifact.method == "temperature":
        # Temperature needs logits; if only probs are available keep them unchanged.
        return np.clip(p, 1e-7, 1.0 - 1e-7)
    if artifact.method in ("isotonic", "temperature_isotonic"):
        p = apply_isotonic_from_thresholds(p, artifact.isotonic_x, artifact.isotonic_y)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def save_calibration(path: str | Path, artifact: CalibrationArtifact, extra: Optional[Dict] = None) -> None:
    payload = artifact.to_dict()
    if extra:
        payload["metrics"] = extra
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path: str | Path) -> CalibrationArtifact:
    with Path(path).open("r") as f:
        payload = json.load(f)
    return CalibrationArtifact.from_dict(payload)
