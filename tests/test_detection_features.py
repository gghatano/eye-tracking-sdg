"""Tests for the detection feature extractor and synthetic adapter (M3-A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_eye_tracking.detection import (
    BLINK,
    FIXATION,
    OTHER,
    SACCADE,
    SAMPLE_COLUMNS,
    validate_samples,
)
from synthetic_eye_tracking.detection.features import (
    FEATURE_NAMES,
    extract_features,
    filter_scored,
)
from synthetic_eye_tracking.detection.synthetic import (
    generate_synthetic_samples,
    rawgaze_to_samples,
)


def _hand_made_samples() -> pd.DataFrame:
    """One series: fixation (still) -> saccade (fast move) -> blink (NaN)."""
    rows = []
    t = 0.0
    dt = 2.0  # 500 Hz
    # Fixation: near-constant position around (1, 1) deg.
    for i in range(12):
        rows.append(("s1", "synthetic", t, 1.0 + 0.01 * (i % 2), 1.0, FIXATION))
        t += dt
    # Saccade: fast linear sweep across ~20 deg.
    for i in range(8):
        rows.append(("s1", "synthetic", t, 1.0 + 2.5 * i, 1.0 + 1.5 * i, SACCADE))
        t += dt
    # Blink: NaN positions.
    for i in range(4):
        rows.append(("s1", "synthetic", t, np.nan, np.nan, BLINK))
        t += dt
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def test_extract_features_shapes_and_finiteness():
    df = _hand_made_samples()
    fs = extract_features(df, window=5)

    n = len(df)
    assert fs.X.shape == (n, len(fs.feature_names))
    assert len(fs.feature_names) == len(FEATURE_NAMES)
    assert fs.y.shape == (n,)
    assert fs.groups.shape == (n,)
    # NaN fill: X must be entirely finite.
    assert np.all(np.isfinite(fs.X))
    # Labels preserved including blink.
    assert set(fs.y) == {FIXATION, SACCADE, BLINK}


def test_features_separate_fixation_and_saccade():
    df = _hand_made_samples()
    fs = extract_features(df, window=5)
    vel = fs.X[:, fs.feature_names.index("velocity")]

    fix_vel = vel[fs.y == FIXATION]
    sac_vel = vel[fs.y == SACCADE]
    # Saccade velocity clearly exceeds fixation velocity.
    assert sac_vel.mean() > 10.0 * max(fix_vel.mean(), 1e-6)
    assert sac_vel.mean() > fix_vel.mean()


def test_filter_scored_drops_other_and_blink():
    df = _hand_made_samples()
    fs = extract_features(df, window=5)
    scored = filter_scored(fs)
    assert set(scored.y) == {FIXATION, SACCADE}
    assert scored.X.shape[0] == int(np.sum(np.isin(fs.y, [FIXATION, SACCADE])))
    assert scored.X.shape[1] == fs.X.shape[1]


def test_extract_features_short_series_no_crash():
    df = pd.DataFrame(
        [("a", "synthetic", 0.0, 1.0, 1.0, FIXATION)], columns=SAMPLE_COLUMNS
    )
    fs = extract_features(df)
    assert fs.X.shape == (1, len(FEATURE_NAMES))
    assert np.all(np.isfinite(fs.X))


def test_multiple_series_processed_independently():
    df = _hand_made_samples()
    df2 = df.copy()
    df2["series_id"] = "s2"
    both = pd.concat([df, df2], ignore_index=True)
    fs = extract_features(both, window=5)
    assert set(fs.groups) == {"s1", "s2"}
    assert fs.X.shape[0] == len(both)


def test_rawgaze_to_samples_hand_made():
    raw = pd.DataFrame(
        {
            "subject_id": ["S0", "S0", "S0"],
            "trial_id": ["S0_T000", "S0_T000", "S0_T000"],
            "stimulus_id": ["ST0", "ST0", "ST0"],
            "timestamp_ms": [0.0, 2.0, 4.0],
            "x_px": [960.0, 1100.0, np.nan],
            "y_px": [540.0, 600.0, np.nan],
            "pupil_size": [3.5, 3.5, np.nan],
            "validity": [1, 1, 0],
            "event_index": [0, 1, 2],
            "event_type": ["fixation", "saccade", "blink"],
            "aoi_id": ["L0_W0", None, None],
        }
    )
    samples = rawgaze_to_samples(
        raw,
        screen_res_px=(1920.0, 1080.0),
        screen_size_m=(0.50, 0.30),
        view_dist_m=0.65,
    )
    assert list(samples.columns) == SAMPLE_COLUMNS
    assert (samples["series_id"] == "syn_S0_S0_T000").all()
    assert (samples["source"] == "synthetic").all()
    assert list(samples["label"]) == [FIXATION, SACCADE, BLINK]
    # Centre pixel -> ~0 deg.
    assert abs(samples["x_deg"].iloc[0]) < 1e-6
    assert abs(samples["y_deg"].iloc[0]) < 1e-6
    # Blink stays NaN.
    assert np.isnan(samples["x_deg"].iloc[2])
    assert np.isnan(samples["y_deg"].iloc[2])


def test_generate_synthetic_samples_end_to_end():
    df = generate_synthetic_samples(
        n_domains=2, n_subjects=2, trials_per_subject=2, n_stimuli=1
    )
    # Valid frame.
    validate_samples(df)
    assert df["series_id"].nunique() > 1
    labels = set(df["label"])
    assert FIXATION in labels
    assert SACCADE in labels

    # Non-blink rows have finite, plausible degree coordinates.
    non_blink = df[df["label"] != BLINK]
    finite = non_blink.dropna(subset=["x_deg", "y_deg"])
    assert len(finite) > 0
    assert np.all(np.isfinite(finite["x_deg"].to_numpy()))
    assert np.all(np.isfinite(finite["y_deg"].to_numpy()))
    assert finite["x_deg"].abs().max() < 60.0
    assert finite["y_deg"].abs().max() < 60.0


def test_generate_synthetic_samples_unique_ids_across_domains():
    df = generate_synthetic_samples(
        n_domains=2, n_subjects=2, trials_per_subject=1, n_stimuli=1
    )
    prefixes = {sid.split("_")[0] for sid in df["series_id"].unique()}
    # Domain prefixes d0 / d1 present.
    assert "d0" in prefixes
    assert "d1" in prefixes
