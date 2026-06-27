"""Sanity tests for the AMEX competition metric.

These tests do NOT validate against published Kaggle leaderboard OOF scores --
that requires a real OOF file in tests/fixtures/ and is handled by a separate
validation script. Here we only assert:

1. perfect predictions => G == 1.0 (M depends on positive density)
2. inverse predictions => G is very negative, fast matches reference
3. uniform random predictions => M is close to 0
4. fast and reference implementations agree on random + realistic fixtures
5. metric is permutation-invariant
6. metric is monotonically non-increasing under added noise
7. metric rejects malformed inputs
8. constant predictions don't crash
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from amex.evaluation.metrics import (
    amex_metric,
    amex_metric_components,
    amex_metric_reference,
)


def _make_labels(n: int = 5_000, pos_rate: float = 0.26, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < pos_rate).astype(np.float64)


def test_perfect_predictions_g_is_one() -> None:
    y_true = _make_labels()
    m, g, _ = amex_metric_components(y_true, y_true)
    assert g == pytest.approx(1.0, abs=1e-9)
    assert 0.5 <= m <= 1.0


def test_random_predictions_score_near_zero() -> None:
    rng = np.random.default_rng(0)
    y_true = _make_labels(seed=1)
    scores = rng.random(y_true.size)
    m = amex_metric(y_true, scores)
    assert abs(m) < 0.05


def test_inverse_predictions_match_reference() -> None:
    """With asymmetric 20:1 weight, inverse predictions can yield G < -1.

    We don't assert exact G == -1 (would be true for symmetric Gini only);
    instead we require strong negativity AND agreement with the slow
    reference implementation.
    """
    y_true = _make_labels(seed=2)
    scores = 1.0 - y_true
    _, g, d = amex_metric_components(y_true, scores)
    assert g < -1.0  # weighted Gini overshoots -1 for inverted predictions
    assert d <= 1e-6
    assert amex_metric(y_true, scores) == pytest.approx(
        amex_metric_reference(y_true, scores), abs=1e-9
    )


def test_fast_matches_reference_random() -> None:
    rng = np.random.default_rng(123)
    y_true = _make_labels(seed=3)
    scores = rng.random(y_true.size)
    assert amex_metric(y_true, scores) == pytest.approx(
        amex_metric_reference(y_true, scores), abs=1e-9
    )


def test_fast_matches_reference_realistic() -> None:
    """Score = signal + noise, signal correlates with truth (typical GBM output)."""
    rng = np.random.default_rng(42)
    y_true = _make_labels(seed=4)
    scores = y_true * 0.6 + rng.normal(0.0, 1.0, size=y_true.size)
    assert amex_metric(y_true, scores) == pytest.approx(
        amex_metric_reference(y_true, scores), abs=1e-9
    )


def test_permutation_invariant() -> None:
    """Shuffling the same (label, score) pairs must not change M."""
    rng = np.random.default_rng(7)
    y_true = _make_labels(seed=5)
    scores = rng.random(y_true.size)
    m_a = amex_metric(y_true, scores)
    order = rng.permutation(y_true.size)
    m_b = amex_metric(y_true[order], scores[order])
    assert m_a == pytest.approx(m_b, abs=1e-9)


def test_monotonic_in_noise() -> None:
    """Adding more noise to perfect predictions degrades M monotonically."""
    rng = np.random.default_rng(11)
    y_true = _make_labels(seed=6)
    base = y_true.astype(np.float64)
    noise_levels = [0.0, 0.1, 0.3, 1.0, 5.0]
    scores_history = [
        amex_metric(y_true, base + rng.normal(0, s, y_true.size)) for s in noise_levels
    ]
    for prev, curr in pairwise(scores_history):
        assert prev >= curr - 1e-3, (noise_levels, scores_history)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        amex_metric([0, 1, 0], [0.5, 0.5])
    with pytest.raises(ValueError, match="0/1"):
        amex_metric([0, 2, 0], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="empty"):
        amex_metric([], [])
    with pytest.raises(ValueError, match="NaN"):
        amex_metric([0, 1], [0.5, float("nan")])


def test_constant_predictions_finite() -> None:
    """All predictions equal: ranking is degenerate but must not crash."""
    y_true = _make_labels(seed=8)
    scores = np.ones_like(y_true)
    m = amex_metric(y_true, scores)
    assert np.isfinite(m)
    assert -2.0 <= m <= 1.0
