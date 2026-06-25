"""Shared contract for the sim-to-real event-detection experiment (M3-A).

Both the real-data loader (Lund2013) and the synthetic adapter must produce a
common *labeled sample frame* so that one feature extractor and one classifier
treat them identically. Working in degrees of visual angle makes features
device-independent, which is what lets a detector trained on synthetic data
transfer to a real recording captured on different hardware.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Labeled sample frame: the common intermediate representation
# --------------------------------------------------------------------------- #

SAMPLE_COLUMNS: list[str] = [
    "series_id",   # unique id per recording/trial (str)
    "source",      # "real" | "synthetic"
    "t_ms",        # sample time in ms (float, monotonic within series)
    "x_deg",       # horizontal gaze position, degrees of visual angle
    "y_deg",       # vertical gaze position, degrees of visual angle
    "label",       # one of LABELS (str); IGNORE rows excluded from train/eval
]

# Event classes the detector predicts. The synthetic generator only produces
# fixation / saccade / blink, so anything else in real data (smooth pursuit,
# PSO, undefined) maps to OTHER and is excluded from training and scoring.
FIXATION = "fixation"
SACCADE = "saccade"
BLINK = "blink"
OTHER = "other"

LABELS: list[str] = [FIXATION, SACCADE, BLINK, OTHER]

# The classes actually scored in the fixation-vs-saccade detection task.
SCORED_LABELS: list[str] = [FIXATION, SACCADE]

# Lund2013 integer label -> our label (1=fix,2=sac,3=PSO,4=pursuit,5=blink,6=undef)
LUND_LABEL_MAP: dict[int, str] = {
    1: FIXATION,
    2: SACCADE,
    3: OTHER,   # PSO (post-saccadic oscillation): not modelled synthetically
    4: OTHER,   # smooth pursuit: not modelled synthetically
    5: BLINK,
    6: OTHER,   # undefined / uncertain
}


def pix_to_deg(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    *,
    screen_res_px: tuple[float, float],
    screen_size_m: tuple[float, float],
    view_dist_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to degrees of visual angle about screen centre.

    Uses the small-angle-correct ``atan`` mapping with the physical pixel pitch
    (screen size / resolution) and the viewing distance.
    """
    x_pix = np.asarray(x_pix, dtype=float)
    y_pix = np.asarray(y_pix, dtype=float)
    res_x, res_y = screen_res_px
    size_x, size_y = screen_size_m
    pitch_x = size_x / res_x  # metres per pixel
    pitch_y = size_y / res_y
    dx_m = (x_pix - res_x / 2.0) * pitch_x
    dy_m = (y_pix - res_y / 2.0) * pitch_y
    x_deg = np.degrees(np.arctan2(dx_m, view_dist_m))
    y_deg = np.degrees(np.arctan2(dy_m, view_dist_m))
    return x_deg, y_deg


def validate_samples(df: pd.DataFrame, *, name: str = "samples") -> pd.DataFrame:
    """Validate and reorder a labeled sample frame to ``SAMPLE_COLUMNS``."""
    have, want = set(df.columns), set(SAMPLE_COLUMNS)
    missing, extra = want - have, have - want
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    bad = set(df["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"{name} has unknown labels: {sorted(bad)}")
    return df[SAMPLE_COLUMNS]
