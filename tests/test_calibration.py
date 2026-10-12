"""Temperature scaling on synthetic logits: recovers a known temperature, never uses labels it should not."""

from __future__ import annotations

import math

import numpy as np
import pytest

from routing_lab.calibration import fit_temperature, negative_log_likelihood


def sample(temperature: float, rows: int = 4000, options: int = 5, seed: int = 0):
    """Logits whose softmax is calibrated only after dividing by ``temperature``."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, size=(rows, options)) * temperature
    probabilities = np.exp(logits / temperature)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = [int(rng.choice(options, p=row)) for row in probabilities]
    return logits.tolist(), targets


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_recovers_the_generating_temperature(true_temperature):
    logits, targets = sample(true_temperature)
    assert fit_temperature(logits, targets) == pytest.approx(true_temperature, rel=0.1)


def test_fit_lowers_calibration_loss_and_supports_mixed_option_counts():
    logits, targets = sample(0.4, rows=1500)
    logits += [[2.0, -1.0, 0.5], [0.1, 0.2]]
    targets += [0, 1]
    fitted = fit_temperature(logits, targets)
    assert negative_log_likelihood(logits, targets, fitted) < negative_log_likelihood(logits, targets, 1.0)


def test_scaling_never_changes_the_chosen_option():
    logits, _ = sample(0.5, rows=50)
    for row in logits:
        assert int(np.argmax(np.asarray(row) / 0.3)) == int(np.argmax(row))


@pytest.mark.parametrize(("logits", "targets"), [
    ([], []), ([[1.0, 2.0]], []), ([[1.0, 2.0]], [2]), ([[1.0, math.nan]], [0]),
])
def test_rejects_unusable_calibration_rows(logits, targets):
    with pytest.raises(ValueError):
        negative_log_likelihood(logits, targets, 1.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf])
def test_rejects_bad_temperatures(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        negative_log_likelihood([[1.0, 2.0]], [0], temperature)


@pytest.mark.parametrize("margin", [2.0, 50.0])
def test_perfectly_separable_data_is_reported_as_degenerate_not_fitted(margin):
    logits = [[margin, -margin]] * 40 + [[-margin, margin]] * 40
    targets = [0] * 40 + [1] * 40
    with pytest.raises(ValueError, match="search bound|perfectly separable"):
        fit_temperature(logits, targets)
