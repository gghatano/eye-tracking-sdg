"""Per-sample windowed feature extraction for event detection (M3-A).

The feature extractor is shared by real and synthetic data so that a single
classifier transfers between domains. To make features device- and
sampling-rate independent we work in **degrees of visual angle** and use
**time-derivative** quantities (velocity / acceleration computed from explicit
time deltas, not sample-index deltas) plus rolling-window summary statistics.

Each ``series_id`` is processed independently, sorted by ``t_ms``. Missing
positions (blinks / data loss, encoded as NaN ``x_deg`` / ``y_deg``) yield NaN
intermediate features which are then filled with 0; an explicit ``is_missing``
flag preserves that information for the classifier. The blink/other labels on
those rows are kept untouched -- filtering to the scored classes is the
caller's responsibility (see :func:`filter_scored`).

The extractor is fully deterministic: no randomness is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import SAMPLE_COLUMNS, SCORED_LABELS

# Ordered list of the features produced per sample. Keep in sync with
# ``_features_for_series``; ``FeatureSet.feature_names`` exposes it.
FEATURE_NAMES: list[str] = [
    "velocity",          # instantaneous speed, deg/s
    "acceleration",      # delta velocity / delta t, deg/s^2
    "win_vel_mean",      # mean velocity over +/- window samples
    "win_vel_std",       # std velocity over the window
    "win_vel_median",    # median velocity over the window
    "win_vel_max",       # max velocity over the window
    "win_disp",          # spatial dispersion: x-range + y-range in the window
    "win_centroid_dist", # distance from sample to window centroid (deg)
    "is_missing",        # 1.0 if the sample position is NaN (blink/data loss)
]


@dataclass(frozen=True)
class FeatureSet:
    """Container for an extracted feature matrix and aligned metadata."""

    X: np.ndarray            # (n_samples, n_features) float
    y: np.ndarray            # (n_samples,) str labels
    groups: np.ndarray       # (n_samples,) str series_id
    feature_names: list[str]


def _features_for_series(
    t_ms: np.ndarray,
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute the per-sample feature matrix for a single, time-sorted series.

    Returns an (n, n_features) array. Robust to very short series (1-2 samples):
    derivative features that are undefined simply become 0.
    """
    n = len(t_ms)
    n_feat = len(FEATURE_NAMES)
    out = np.zeros((n, n_feat), dtype=float)

    is_missing = ~np.isfinite(x_deg) | ~np.isfinite(y_deg)
    out[:, FEATURE_NAMES.index("is_missing")] = is_missing.astype(float)

    if n < 2:
        # Single sample: no derivatives, no window -> all-zero (plus is_missing).
        return out

    # Instantaneous velocity (deg/s). Position delta / time delta between
    # consecutive samples. Velocity[i] is the speed arriving at sample i; the
    # first sample has no predecessor -> 0.
    dt = np.diff(t_ms) / 1000.0  # seconds
    dt = np.where(dt > 0, dt, np.nan)  # guard against zero / duplicate stamps
    dx = np.diff(x_deg)
    dy = np.diff(y_deg)
    step = np.hypot(dx, dy)
    vel = np.empty(n, dtype=float)
    vel[0] = np.nan
    vel[1:] = step / dt

    # Acceleration (deg/s^2): delta velocity / delta t. Aligned to sample i.
    # The first two samples have no well-defined acceleration -> NaN (-> 0).
    acc = np.full(n, np.nan, dtype=float)
    if n >= 3:
        acc[2:] = (vel[2:] - vel[1:-1]) / dt[1:]

    out[:, FEATURE_NAMES.index("velocity")] = vel
    out[:, FEATURE_NAMES.index("acceleration")] = acc

    # Rolling-window statistics over +/- ``window`` samples (inclusive).
    i_vel_mean = FEATURE_NAMES.index("win_vel_mean")
    i_vel_std = FEATURE_NAMES.index("win_vel_std")
    i_vel_med = FEATURE_NAMES.index("win_vel_median")
    i_vel_max = FEATURE_NAMES.index("win_vel_max")
    i_disp = FEATURE_NAMES.index("win_disp")
    i_cdist = FEATURE_NAMES.index("win_centroid_dist")

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)

        v_win = vel[lo:hi]
        v_valid = v_win[np.isfinite(v_win)]
        if v_valid.size:
            out[i, i_vel_mean] = float(np.mean(v_valid))
            out[i, i_vel_std] = float(np.std(v_valid))
            out[i, i_vel_med] = float(np.median(v_valid))
            out[i, i_vel_max] = float(np.max(v_valid))

        xw = x_deg[lo:hi]
        yw = y_deg[lo:hi]
        xw_v = xw[np.isfinite(xw)]
        yw_v = yw[np.isfinite(yw)]
        if xw_v.size and yw_v.size:
            x_range = float(np.max(xw_v) - np.min(xw_v))
            y_range = float(np.max(yw_v) - np.min(yw_v))
            out[i, i_disp] = x_range + y_range
            cx = float(np.mean(xw_v))
            cy = float(np.mean(yw_v))
            if np.isfinite(x_deg[i]) and np.isfinite(y_deg[i]):
                out[i, i_cdist] = float(np.hypot(x_deg[i] - cx, y_deg[i] - cy))

    # Fill any remaining NaNs (e.g. first-sample derivatives, blink rows) with 0.
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def extract_features(samples_df: pd.DataFrame, *, window: int = 7) -> FeatureSet:
    """Extract per-sample windowed features from a labeled sample frame.

    ``samples_df`` must conform to ``SAMPLE_COLUMNS``. Each ``series_id`` is
    processed independently, sorted by ``t_ms``. ALL rows are returned with
    their labels (including OTHER and BLINK); use :func:`filter_scored` to
    restrict to the fixation-vs-saccade task.

    Parameters
    ----------
    samples_df:
        Labeled sample frame.
    window:
        Half-width (in samples) of the rolling statistics window; the full
        window spans ``2 * window + 1`` samples centred on each sample.
    """
    missing = set(SAMPLE_COLUMNS) - set(samples_df.columns)
    if missing:
        raise ValueError(f"samples_df missing columns: {sorted(missing)}")

    if len(samples_df) == 0:
        return FeatureSet(
            X=np.zeros((0, len(FEATURE_NAMES)), dtype=float),
            y=np.empty(0, dtype=object),
            groups=np.empty(0, dtype=object),
            feature_names=list(FEATURE_NAMES),
        )

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []

    # Stable per-series processing; preserve first-appearance order of series.
    for series_id, g in samples_df.groupby("series_id", sort=False):
        g = g.sort_values("t_ms", kind="stable")
        t_ms = g["t_ms"].to_numpy(dtype=float)
        x_deg = g["x_deg"].to_numpy(dtype=float)
        y_deg = g["y_deg"].to_numpy(dtype=float)
        feats = _features_for_series(t_ms, x_deg, y_deg, window)
        X_parts.append(feats)
        y_parts.append(g["label"].to_numpy())
        g_parts.append(np.asarray([series_id] * len(g), dtype=object))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts).astype(object)
    groups = np.concatenate(g_parts).astype(object)
    return FeatureSet(X=X, y=y, groups=groups, feature_names=list(FEATURE_NAMES))


def filter_scored(fs: FeatureSet, labels: list[str] = SCORED_LABELS) -> FeatureSet:
    """Return a new ``FeatureSet`` keeping only rows whose label is in ``labels``.

    Used to build the fixation-vs-saccade training/eval task by dropping OTHER
    and BLINK rows.
    """
    keep = np.isin(fs.y, list(labels))
    return FeatureSet(
        X=fs.X[keep],
        y=fs.y[keep],
        groups=fs.groups[keep],
        feature_names=list(fs.feature_names),
    )
