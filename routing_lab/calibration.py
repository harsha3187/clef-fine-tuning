"""Temperature scaling for choice logits, fitted only on held-out calibration data."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

TEMPERATURE_BOUNDS = (0.05, 20.0)


def negative_log_likelihood(
    logits: Sequence[Sequence[float]], targets: Sequence[int], temperature: float,
) -> float:
    """Mean cross-entropy of the temperature-scaled softmax; each row may have its own option count."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite")
    if not logits or len(logits) != len(targets):
        raise ValueError("Calibration needs one target per logit row, and at least one row")
    total = 0.0
    for row, target in zip(logits, targets, strict=True):
        scaled = np.asarray(row, dtype=np.float64) / temperature
        if not 0 <= target < scaled.size or not np.isfinite(scaled).all():
            raise ValueError("Calibration row has a bad target index or non-finite logit")
        # log(1 + sum_o exp(s_o - s_target)): stays accurate when the loss is tiny or the margins huge.
        others = np.delete(scaled, target) - scaled[target]
        total += float(np.logaddexp(0.0, np.logaddexp.reduce(others))) if others.size else 0.0
    return total / len(targets)


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Return the single temperature minimizing calibration NLL; accuracy is unchanged by it."""
    low, high = TEMPERATURE_BOUNDS
    result = minimize_scalar(
        lambda log_t: negative_log_likelihood(logits, targets, math.exp(log_t)),
        bounds=(math.log(low), math.log(high)), method="bounded", options={"xatol": 1e-4},
    )
    temperature = float(math.exp(result.x))
    if temperature <= low * 1.01 or temperature >= high / 1.01:
        raise ValueError("Fitted temperature hit its search bound; the calibration data is degenerate")
    if negative_log_likelihood(logits, targets, temperature) < 1e-6:
        raise ValueError("Calibration loss is effectively zero; the data is perfectly separable, not calibratable")
    return temperature
