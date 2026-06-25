"""Tests for the rule-based generator (spec section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic_eye_tracking.aoi import build_reading_grid
from synthetic_eye_tracking.config import Config, DistributionSpec, ExperimentConfig
from synthetic_eye_tracking.events import (
    sample_distribution,
    saccade_amplitude,
    saccade_angle_deg,
    saccade_velocity_px_s,
)
from synthetic_eye_tracking.generators.rule_based import RuleBasedGenerator
from synthetic_eye_tracking.schema import (
    AOI_LAYOUT_COLUMNS,
    PARTICIPANT_COLUMNS,
    STIMULUS_COLUMNS,
    EventType,
    validate_aoi_summary,
    validate_events,
    validate_raw_gaze,
)


def small_config(seed: int = 1) -> Config:
    return Config(
        experiment=ExperimentConfig(
            n_subjects=3, n_stimuli=2, trials_per_subject=2, seed=seed
        )
    )


@pytest.fixture
def gen() -> RuleBasedGenerator:
    return RuleBasedGenerator(small_config())


# --------------------------------------------------------------------------- #
# events.py helpers
# --------------------------------------------------------------------------- #
def test_sample_distribution_truncation():
    rng = np.random.default_rng(0)
    spec = DistributionSpec(distribution="lognormal", mean=240, sd=80, min=80, max=800)
    vals = sample_distribution(spec, 1000, rng)
    assert vals.shape == (1000,)
    assert vals.min() >= 80 - 1e-9
    assert vals.max() <= 800 + 1e-9


def test_sample_distribution_families():
    rng = np.random.default_rng(0)
    for dist in ("normal", "lognormal", "gamma"):
        spec = DistributionSpec(distribution=dist, mean=100, sd=20, min=0, max=500)
        vals = sample_distribution(spec, 500, rng)
        assert np.all(np.isfinite(vals))


def test_saccade_geometry():
    assert saccade_amplitude((0, 0), (3, 4)) == pytest.approx(5.0)
    assert saccade_angle_deg((0, 0), (1, 0)) == pytest.approx(0.0)
    assert saccade_angle_deg((0, 0), (0, 1)) == pytest.approx(90.0)
    assert saccade_velocity_px_s(100, 50) == pytest.approx(2000.0)


# --------------------------------------------------------------------------- #
# aoi / participants / stimuli
# --------------------------------------------------------------------------- #
def test_aoi_layout_schema_and_order(gen):
    layout = gen.aoi_layout
    assert list(layout.columns) == AOI_LAYOUT_COLUMNS
    s = gen.config.stimulus
    per_stim = s.n_lines * s.words_per_line
    assert len(layout) == per_stim * gen.config.experiment.n_stimuli
    # difficulty in [0, 1]
    assert layout["difficulty"].between(0, 1).all()
    # left->right, top->bottom: word 0 of a line is leftmost; lines descend.
    one = layout[layout["stimulus_id"] == "ST000"]
    l0 = one[one["line_index"] == 0].sort_values("word_index")
    assert l0["center_x"].is_monotonic_increasing
    assert one.groupby("line_index")["center_y"].first().is_monotonic_increasing
    assert (one["aoi_id"] == "L0_W0").any()


def test_participants_schema(gen):
    p = gen.participants
    assert list(p.columns) == PARTICIPANT_COLUMNS
    assert len(p) == 3
    assert (p["fixation_noise_sd"] > 0).all()
    assert p["missing_rate"].between(0, 1).all()
    assert p["regression_tendency"].between(0, 1).all()
    assert p["skip_tendency"].between(0, 1).all()
    assert (p["pupil_baseline"] > 0).all()
    assert list(p["subject_id"]) == ["S000", "S001", "S002"]


def test_stimuli_schema(gen):
    st = gen.stimuli
    assert list(st.columns) == STIMULUS_COLUMNS
    assert len(st) == 2
    assert st["difficulty"].between(0, 1).all()
    assert list(st["stimulus_id"]) == ["ST000", "ST001"]


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def test_events_schema_and_nonempty(gen):
    events = gen.generate_events()
    validate_events(events)  # raises on mismatch
    assert list(events.columns) == [
        c for c in events.columns
    ]  # column set verified by validate
    assert len(events) > 0
    types = set(events["event_type"].unique())
    assert EventType.FIXATION.value in types
    assert EventType.SACCADE.value in types


def test_fixation_durations_in_bounds(gen):
    events = gen.generate_events()
    spec = gen.config.event_distribution.fixation_duration_ms
    fix = events[events["event_type"] == EventType.FIXATION.value]
    assert (fix["duration_ms"] >= spec.min - 1e-6).all()
    assert (fix["duration_ms"] <= spec.max + 1e-6).all()


def test_event_times_monotonic_per_trial(gen):
    events = gen.generate_events()
    for _, g in events.groupby(["subject_id", "trial_id"], sort=False):
        g = g.sort_values("event_index")
        # end >= start, duration consistent
        assert (g["end_time_ms"] >= g["start_time_ms"]).all()
        assert np.allclose(
            g["duration_ms"], g["end_time_ms"] - g["start_time_ms"]
        )
        # start times non-decreasing across event_index
        assert g["start_time_ms"].is_monotonic_increasing
        # event_index contiguous from 0
        assert list(g["event_index"]) == list(range(len(g)))


def test_aoi_id_only_on_fixations(gen):
    events = gen.generate_events()
    non_fix = events[events["event_type"] != EventType.FIXATION.value]
    assert non_fix["aoi_id"].isna().all()
    fix = events[events["event_type"] == EventType.FIXATION.value]
    assert fix["aoi_id"].notna().all()


# --------------------------------------------------------------------------- #
# raw gaze
# --------------------------------------------------------------------------- #
def test_raw_gaze_schema_and_spacing(gen):
    events = gen.generate_events()
    raw = gen.generate_raw_gaze(events)
    validate_raw_gaze(raw)
    assert len(raw) > 0
    dt = 1000.0 / gen.config.sampling.frequency_hz
    for _, g in raw.groupby(["subject_id", "trial_id"], sort=False):
        ts = g["timestamp_ms"].to_numpy()
        assert np.all(np.diff(ts) >= -1e-6)  # monotonic
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if len(diffs):
            assert np.median(diffs) == pytest.approx(dt, rel=0.05)


def test_blink_rows_have_nan_xy(gen):
    events = gen.generate_events()
    raw = gen.generate_raw_gaze(events)
    blink = raw[raw["event_type"] == EventType.BLINK.value]
    if len(blink):
        assert blink["x_px"].isna().all()
        assert blink["y_px"].isna().all()
        assert blink["pupil_size"].isna().all()
        assert (blink["validity"] == 0).all()


def test_raw_gaze_requires_events_first():
    g = RuleBasedGenerator(small_config())
    with pytest.raises(RuntimeError):
        g.generate_raw_gaze(pd.DataFrame())


# --------------------------------------------------------------------------- #
# aoi summary
# --------------------------------------------------------------------------- #
def test_aoi_summary_schema_and_counts(gen):
    events = gen.generate_events()
    summary = gen.generate_aoi_summary(events)
    validate_aoi_summary(summary)
    assert len(summary) > 0

    # fixation_count must equal number of fixation events per (trial, aoi).
    fix = events[
        (events["event_type"] == EventType.FIXATION.value)
        & (events["aoi_id"].notna())
    ]
    expected = (
        fix.groupby(["subject_id", "trial_id", "stimulus_id", "aoi_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    merged = summary.merge(
        expected, on=["subject_id", "trial_id", "stimulus_id", "aoi_id"]
    )
    assert (merged["fixation_count"] == merged["n"]).all()
    assert len(merged) == len(summary)
    # total >= first fixation duration
    assert (
        summary["total_fixation_duration_ms"]
        >= summary["first_fixation_duration_ms"] - 1e-6
    ).all()
    assert (summary["time_to_first_fixation_ms"] >= -1e-6).all()


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_seed():
    g1 = RuleBasedGenerator(small_config(seed=7))
    e1 = g1.generate_events()
    r1 = g1.generate_raw_gaze(e1)
    a1 = g1.generate_aoi_summary(e1)

    g2 = RuleBasedGenerator(small_config(seed=7))
    e2 = g2.generate_events()
    r2 = g2.generate_raw_gaze(e2)
    a2 = g2.generate_aoi_summary(e2)

    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(g1.participants, g2.participants)
    pd.testing.assert_frame_equal(g1.stimuli, g2.stimuli)
    pd.testing.assert_frame_equal(g1.aoi_layout, g2.aoi_layout)


def test_different_seed_differs():
    g1 = RuleBasedGenerator(small_config(seed=1))
    g2 = RuleBasedGenerator(small_config(seed=2))
    e1 = g1.generate_events()
    e2 = g2.generate_events()
    # Highly unlikely to be identical with different seeds.
    assert not e1.equals(e2)
