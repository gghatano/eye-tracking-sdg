"""Canonical data schema shared by every generator and the evaluator.

This module is the contract: rule-based and model-based generators MUST emit
DataFrames with exactly these columns (section 14: "same schema"), and the
evaluator reads them back. Column order is normative so CSV outputs are stable.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    BLINK = "blink"


class TransitionType(str, Enum):
    """Relative AOI transition categories (Markov model state space, 7.4)."""

    NEXT = "next"
    SKIP = "skip"
    REGRESSION_1 = "regression_1"
    REGRESSION_2PLUS = "regression_2plus"
    LINE_BREAK = "line_break"
    END = "end"


# --------------------------------------------------------------------------- #
# Column definitions (normative order)
# --------------------------------------------------------------------------- #

EVENT_COLUMNS: list[str] = [
    "subject_id",
    "trial_id",
    "stimulus_id",
    "event_index",
    "event_type",
    "start_time_ms",
    "end_time_ms",
    "duration_ms",
    "x_px",
    "y_px",
    "aoi_id",
    "saccade_amp_px",
    "saccade_angle_deg",
    "saccade_velocity_px_s",
    "validity",
]

RAW_GAZE_COLUMNS: list[str] = [
    "subject_id",
    "trial_id",
    "stimulus_id",
    "timestamp_ms",
    "x_px",
    "y_px",
    "pupil_size",
    "validity",
    "event_index",
    "event_type",
    "aoi_id",
]

AOI_SUMMARY_COLUMNS: list[str] = [
    "subject_id",
    "trial_id",
    "stimulus_id",
    "aoi_id",
    "first_fixation_duration_ms",
    "total_fixation_duration_ms",
    "fixation_count",
    "visit_count",
    "time_to_first_fixation_ms",
    "regression_count",
]

PARTICIPANT_COLUMNS: list[str] = [
    "subject_id",
    "reading_speed_factor",
    "fixation_noise_sd",
    "regression_tendency",
    "skip_tendency",
    "missing_rate",
    "pupil_baseline",
]

STIMULUS_COLUMNS: list[str] = [
    "stimulus_id",
    "layout_type",
    "n_lines",
    "words_per_line",
    "n_aoi",
    "difficulty",
]

AOI_LAYOUT_COLUMNS: list[str] = [
    "stimulus_id",
    "aoi_id",
    "line_index",
    "word_index",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "center_x",
    "center_y",
    "difficulty",
]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the given columns (stable for writers)."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def ensure_columns(df: pd.DataFrame, columns: list[str], *, name: str = "frame") -> pd.DataFrame:
    """Validate that ``df`` has exactly ``columns`` and return it reordered.

    Raises ValueError listing missing/extra columns so generator bugs surface
    immediately rather than producing silently-malformed CSVs.
    """
    have = set(df.columns)
    want = set(columns)
    missing = want - have
    extra = have - want
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    return df[columns]


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_columns(df, EVENT_COLUMNS, name="events")


def validate_raw_gaze(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_columns(df, RAW_GAZE_COLUMNS, name="raw_gaze")


def validate_aoi_summary(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_columns(df, AOI_SUMMARY_COLUMNS, name="aoi_summary")
