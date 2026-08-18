"""Phase 1 evaluation metrics: MAE, MSE, RMSE, MAPE, NRMSE-per-horizon, VPT.

Every function takes numpy arrays `y_true`, `y_pred`, `mask` of identical
shape. `mask=True` marks a cell that was ORIGINALLY OBSERVED (never
IDW-imputed) — cells where `mask=False` are excluded from every metric
here. Never pass IDW-imputed values through these functions as if they
were ground truth; filter with the real observation mask first (these
functions do that filtering internally, given the mask).

As of the first real A100-ready baseline (LiteSTGNN + SAWS temperature,
96h in / 48h out), the project has explicitly settled the definitions
that were previously flagged UNRESOLVED during the local smoke test:

- MAPE: targets with |y_true| < 1.0 degC are excluded (in addition to
  the observation mask), per explicit project decision — not the
  smoke-test's eps=1e-3. MAPE remains secondary/diagnostic only: it must
  never be used for training, early stopping, model selection, or
  hyperparameter tuning. `mape_exclusion_stats` reports how many/what
  percentage of otherwise-valid targets this rule drops, so the
  resulting MAPE can be read alongside its own exclusion rate.

- NRMSE: `station_normalized_nrmse_per_horizon` implements the
  project's own definition — NRMSE(t_f) = sqrt(mean over valid
  (sample, station) pairs of (y-yhat)^2 / sigma_i^2), where sigma_i is
  the TRAIN-split, originally-observed-only standard deviation for
  station i (the same value used for per-station normalization, see
  PerStationScaler in src/data_prep/saws_dataset.py). This supersedes
  the smoke test's scalar-std NRMSE fallback.

- VPT: epsilon=1.0 is now an explicit project decision (see
  configs/litestgnn_saws_baseline.json's "vpt_epsilon"), not a guess.
  `valid_prediction_time` still takes epsilon as a required argument
  (no hardcoded default) so the function itself stays generic/reusable
  and the decision stays visible at every call site.

The original scalar-std `nrmse_per_horizon` and MAPE eps=1e-3 default
below are kept only because the smoke test still references them — do
not use them for the real SAWS-temperature pipeline.
"""

import numpy as np

MAPE_EPS_NOTE = (
    "eps reused from src/evaluation/evaluate_litestgnn.py's existing "
    "masked_mape (eps=1e-3, exclusion-based). MAPE-for-Celsius-temperature "
    "is conceptually shaky regardless of eps — flagged, not resolved."
)

NRMSE_DEFINITION_NOTE = (
    "NRMSE(t) = RMSE(t) / norm_std, norm_std supplied by caller. Follows "
    "common dynamical-systems/VPT literature convention (matches "
    "analysis/README.md's Lyapunov/VPT framing) but is NOT verified "
    "against a written project proposal - none exists in this repo."
)

STATION_NRMSE_DEFINITION_NOTE = (
    "NRMSE(t_f) = sqrt(mean over valid (sample, station) pairs of "
    "(y - yhat)^2 / sigma_i^2), sigma_i = train-split, "
    "originally-observed-only std for station i (same value used by "
    "PerStationScaler). Explicit project decision, not a placeholder."
)

VPT_EPSILON_NOTE = (
    "VPT epsilon has no value anywhere in this repo or any proposal "
    "document. valid_prediction_time() requires it explicitly with no "
    "default - never call it with a guessed epsilon for reported results."
)

VPT_DEFINITION_NOTE = (
    "VPT = max t such that NRMSE(tau) < epsilon for every tau <= t (the "
    "last CONTINUOUSLY valid lead, not just the last lead below "
    "epsilon). Right-censored (vpt >= final lead) if NRMSE never reaches "
    "epsilon within the evaluated horizon. epsilon=1.0 is an explicit "
    "project decision for the first baseline; the function still "
    "requires epsilon explicitly rather than defaulting to it."
)

MAPE_EXCLUSION_NOTE = (
    "Targets with |y_true| < abs_below (project decision: 1.0 degC) are "
    "excluded from MAPE in addition to the observation mask. MAPE is "
    "secondary/diagnostic only - never used for training, early "
    "stopping, model selection, or hyperparameter tuning."
)


def _masked_values(y_true, y_pred, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return np.array([]), np.array([])
    return np.asarray(y_true)[mask], np.asarray(y_pred)[mask]


def masked_mae(y_true, y_pred, mask):
    t, p = _masked_values(y_true, y_pred, mask)
    return float(np.mean(np.abs(p - t))) if t.size else float("nan")


def masked_mse(y_true, y_pred, mask):
    t, p = _masked_values(y_true, y_pred, mask)
    return float(np.mean((p - t) ** 2)) if t.size else float("nan")


def masked_rmse(y_true, y_pred, mask):
    mse = masked_mse(y_true, y_pred, mask)
    return float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")


def masked_mape(y_true, y_pred, mask, eps=1e-3):
    """Mean absolute percentage error. See MAPE_EPS_NOTE above.

    Cells are excluded (not clipped) when |y_true| <= eps, in addition to
    the observation mask, matching the existing evaluate_litestgnn.py
    convention this is copied from.
    """
    mask = np.asarray(mask, dtype=bool)
    y_true = np.asarray(y_true)
    valid = mask & (np.abs(y_true) > eps)
    if valid.sum() == 0:
        return float("nan")
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_pred[valid] - y_true[valid]) / y_true[valid])) * 100)


def mape_exclusion_stats(y_true, mask, abs_below=1.0):
    """How many/what % of originally-observed (mask=True) targets are
    additionally dropped from MAPE by the |y_true| < abs_below rule. See
    MAPE_EXCLUSION_NOTE. Report this alongside any MAPE number.
    """
    mask = np.asarray(mask, dtype=bool)
    y_true = np.asarray(y_true)
    total_masked_valid = int(mask.sum())
    excluded = int((mask & (np.abs(y_true) < abs_below)).sum())
    pct = (100.0 * excluded / total_masked_valid) if total_masked_valid else float("nan")
    return {
        "abs_below_threshold_degC": abs_below,
        "total_masked_valid_count": total_masked_valid,
        "excluded_count": excluded,
        "excluded_pct": pct,
    }


def station_normalized_nrmse_per_horizon(y_true, y_pred, mask, station_std):
    """Project-defined NRMSE per lead time. See STATION_NRMSE_DEFINITION_NOTE.

    y_true, y_pred, mask: [n_samples, horizon, n_nodes]. station_std:
    [n_nodes] array of per-station train-observed-only standard
    deviations (the same values PerStationScaler was fit with) -
    required, not computed here.
    """
    station_std = np.asarray(station_std, dtype=np.float64)
    if station_std.ndim != 1 or np.any(station_std <= 0):
        raise ValueError("station_std must be a 1D array of positive per-station standard deviations")
    if station_std.shape[0] != y_true.shape[2]:
        raise ValueError(
            f"station_std has {station_std.shape[0]} entries but y_true has "
            f"{y_true.shape[2]} stations - these must match"
        )

    horizon = y_true.shape[1]
    out = np.full(horizon, np.nan, dtype=np.float64)
    variance = station_std ** 2  # [n_nodes]
    for h in range(horizon):
        mask_h = np.asarray(mask[:, h], dtype=bool)  # [n_samples, n_nodes]
        if not mask_h.any():
            continue
        sq_error = (np.asarray(y_pred[:, h]) - np.asarray(y_true[:, h])) ** 2  # [n_samples, n_nodes]
        normalized = sq_error / variance[None, :]
        out[h] = np.sqrt(normalized[mask_h].mean())
    return out


def nrmse_per_horizon(y_true, y_pred, mask, norm_std):
    """RMSE / norm_std at each lead-time step. See NRMSE_DEFINITION_NOTE.

    y_true, y_pred, mask: [n_samples, horizon, n_nodes]. norm_std: a
    single positive float, required, not computed internally.
    """
    if norm_std is None or norm_std <= 0:
        raise ValueError(
            "norm_std must be an explicit positive float supplied by the "
            "caller - see NRMSE_DEFINITION_NOTE, this is not guessed here."
        )
    horizon = y_true.shape[1]
    out = np.full(horizon, np.nan, dtype=np.float64)
    for h in range(horizon):
        out[h] = masked_rmse(y_true[:, h], y_pred[:, h], mask[:, h]) / norm_std
    return out


def valid_prediction_time(nrmse_by_horizon, epsilon, lead_times=None):
    """VPT = max t such that NRMSE(tau) < epsilon for every tau <= t.
    See VPT_DEFINITION_NOTE - this is the last CONTINUOUSLY valid lead,
    not merely the last lead below epsilon (one dip back below epsilon
    after an earlier breach does not extend VPT).

    epsilon is required with no default (see VPT_EPSILON_NOTE) - pass it
    explicitly (e.g. from config["vpt_epsilon"]) rather than relying on a
    library default.

    Returns a dict: {"vpt_hours": float, "right_censored": bool}.
    right_censored=True means NRMSE never reached epsilon within the
    evaluated horizon, i.e. VPT >= the final evaluated lead time.
    """
    if epsilon is None:
        raise ValueError(
            "epsilon is required and has no default - VPT epsilon is not "
            "yet defined by any project proposal or config. See "
            "VPT_EPSILON_NOTE."
        )
    nrmse_by_horizon = np.asarray(nrmse_by_horizon, dtype=np.float64)
    lead_times = (
        np.arange(1, len(nrmse_by_horizon) + 1) if lead_times is None else np.asarray(lead_times)
    )
    invalid = np.where(nrmse_by_horizon >= epsilon)[0]
    if len(invalid) == 0:
        return {"vpt_hours": float(lead_times[-1]), "right_censored": True}
    first_invalid_idx = int(invalid[0])
    if first_invalid_idx == 0:
        return {"vpt_hours": 0.0, "right_censored": False}
    return {"vpt_hours": float(lead_times[first_invalid_idx - 1]), "right_censored": False}
