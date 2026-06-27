"""AMEX Default Prediction competition metric.

The official competition metric (Kaggle 2022) is:

    M = 0.5 * (G + D)

where

- ``G`` is the normalized weighted Gini coefficient, and
- ``D`` is the default rate captured at the top 4 percent of (weighted)
  predictions.

Both ``G`` and ``D`` weight the **negative** class by 20x and the positive
class by 1x. The 4-percent cutoff is measured against the *cumulative weight*,
not the row count: with ~26% positive rate, the top 4% by weight corresponds
to a different (and much smaller) row count than 4% of rows.

This module exports:

- ``amex_metric(y_true, y_pred)``: scalar M, numpy-only fast implementation
- ``amex_metric_components(y_true, y_pred)``: returns ``(M, G, D)`` for diagnostics
- ``amex_metric_reference(y_true, y_pred)``: pandas reference implementation
  used in tests to cross-check the fast path

The two implementations must agree to within ~1e-9 on every fixture; if they
diverge, the fast path is incorrect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

NEG_WEIGHT: float = 20.0
POS_WEIGHT: float = 1.0
CAPTURE_FRACTION: float = 0.04


def _as_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize inputs to 1-D float arrays, validating shape and content."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        msg = f"y_true shape {yt.shape} != y_pred shape {yp.shape}"
        raise ValueError(msg)
    if yt.size == 0:
        msg = "AMEX metric is undefined on empty input"
        raise ValueError(msg)
    uniques = np.unique(yt)
    if not np.all(np.isin(uniques, (0.0, 1.0))):
        msg = f"y_true must contain only 0/1 labels; got {uniques!r}"
        raise ValueError(msg)
    if np.any(np.isnan(yp)):
        msg = "y_pred contains NaN"
        raise ValueError(msg)
    return yt, yp


def _sort_descending(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sort by prediction descending. Ties broken by stable sort on indices."""
    # Negate so argsort ascending equals descending sort.
    order = np.argsort(-y_pred, kind="stable")
    return y_true[order], y_pred[order]


def _weighted_gini(y_true_sorted: NDArray[np.float64]) -> float:
    """Compute the weighted Gini area for predictions already sorted desc.

    Walks down the sorted list, accumulating the cumulative share of positives
    ("lorentz") versus the cumulative share of total weight ("random uniform");
    integrates (lorentz - random) * weight.
    """
    weight = np.where(y_true_sorted == 0.0, NEG_WEIGHT, POS_WEIGHT)
    weight_sum = float(weight.sum())
    if weight_sum == 0.0:
        return 0.0

    pos_weight = y_true_sorted * weight
    total_pos = float(pos_weight.sum())
    if total_pos == 0.0:
        return 0.0

    cum_random = np.cumsum(weight) / weight_sum
    cum_lorentz = np.cumsum(pos_weight) / total_pos
    return float(((cum_lorentz - cum_random) * weight).sum())


def _normalized_weighted_gini(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Normalize the model Gini area by the perfect-ranking Gini area.

    Note: this is the canonical Rohan Rao formulation. With the asymmetric
    20:1 weight, the perfect-ranking denominator does NOT bound the numerator
    symmetrically -- inverse predictions can yield normalized G < -1.
    """
    yt_model, _ = _sort_descending(y_true, y_pred)
    yt_perfect, _ = _sort_descending(y_true, y_true)  # rank by true label desc
    model_gini = _weighted_gini(yt_model)
    perfect_gini = _weighted_gini(yt_perfect)
    if perfect_gini == 0.0:
        return 0.0
    return model_gini / perfect_gini


def _top_four_percent_captured(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Fraction of positive labels captured in the top 4% by *cumulative weight*."""
    yt_sorted, _ = _sort_descending(y_true, y_pred)
    weight = np.where(yt_sorted == 0.0, NEG_WEIGHT, POS_WEIGHT)
    total_weight = float(weight.sum())
    cutoff = CAPTURE_FRACTION * total_weight

    cum_weight = np.cumsum(weight)
    inside = cum_weight <= cutoff
    captured_pos = float(yt_sorted[inside].sum())
    total_pos = float(yt_sorted.sum())
    if total_pos == 0.0:
        return 0.0
    return captured_pos / total_pos


def amex_metric_components(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float, float]:
    """Return ``(M, G, D)`` for diagnostics.

    Use ``amex_metric`` for the scalar score only.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    g = _normalized_weighted_gini(yt, yp)
    d = _top_four_percent_captured(yt, yp)
    m = 0.5 * (g + d)
    return m, g, d


def amex_metric(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the AMEX competition metric ``M = 0.5 * (G + D)``.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Binary default labels, one per customer.
    y_pred : array-like of float
        Predicted default scores (higher == more likely default). Need not be
        probabilities; only the ranking matters.

    Returns
    -------
    float
        Competition score. Perfect ranking returns ~1.0; uniform random ~0.
        Pathologically bad predictions can return negative values.
    """
    m, _, _ = amex_metric_components(y_true, y_pred)
    return m


# ----------------------------------------------------------------------
# Reference implementation: pandas / row-wise. Slower but easy to audit.
# Kept solely so tests can cross-check the fast path.
# ----------------------------------------------------------------------
def amex_metric_reference(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Slow pandas implementation used as a ground-truth oracle in tests."""
    yt, yp = _as_arrays(y_true, y_pred)
    model = pd.DataFrame({"target": yt, "prediction": yp})
    model = model.sort_values("prediction", ascending=False, kind="stable").reset_index(drop=True)
    model["weight"] = model["target"].map({0.0: NEG_WEIGHT, 1.0: POS_WEIGHT})

    # --- D: top 4% captured ---
    cutoff = CAPTURE_FRACTION * model["weight"].sum()
    model["wcum"] = model["weight"].cumsum()
    captured = model.loc[model["wcum"] <= cutoff, "target"].sum()
    total_pos = model["target"].sum()
    d = float(captured / total_pos) if total_pos > 0 else 0.0

    # --- G: normalized weighted Gini ---
    def _gini(frame: pd.DataFrame) -> float:
        w = frame["weight"].to_numpy()
        t = frame["target"].to_numpy()
        cum_random = np.cumsum(w) / w.sum()
        pos_w = t * w
        total_pos_w = pos_w.sum()
        if total_pos_w == 0:
            return 0.0
        cum_lorentz = np.cumsum(pos_w) / total_pos_w
        return float(((cum_lorentz - cum_random) * w).sum())

    perfect = model.sort_values("target", ascending=False, kind="stable").reset_index(drop=True)
    perfect["weight"] = perfect["target"].map({0.0: NEG_WEIGHT, 1.0: POS_WEIGHT})

    g_num = _gini(model)
    g_den = _gini(perfect)
    g = float(g_num / g_den) if g_den != 0 else 0.0

    return 0.5 * (g + d)
