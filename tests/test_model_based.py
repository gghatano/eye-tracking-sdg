"""Tests for the model-based generator (spec section 7, M2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic_eye_tracking.config import Config, ExperimentConfig
from synthetic_eye_tracking.generators.model_based import ModelBasedGenerator
from synthetic_eye_tracking.generators.rule_based import RuleBasedGenerator
from synthetic_eye_tracking.models.markov import (
    STATES,
    MarkovTransitionModel,
    classify_transition,
    parse_aoi,
)
from synthetic_eye_tracking.models.parametric import ParametricModel
from synthetic_eye_tracking.schema import (
    EventType,
    TransitionType,
    validate_aoi_summary,
    validate_events,
    validate_raw_gaze,
)


def ref_config(seed: int = 7) -> Config:
    return Config(
        experiment=ExperimentConfig(
            n_subjects=8, n_stimuli=3, trials_per_subject=4, seed=seed
        )
    )


def gen_config(seed: int = 11) -> Config:
    return Config(
        experiment=ExperimentConfig(
            n_subjects=4, n_stimuli=2, trials_per_subject=3, seed=seed
        )
    )


@pytest.fixture(scope="module")
def reference_events() -> pd.DataFrame:
    return RuleBasedGenerator(ref_config()).generate_events()


@pytest.fixture
def fitted_gen(reference_events) -> ModelBasedGenerator:
    g = ModelBasedGenerator(gen_config())
    g.fit(reference_events)
    return g


# --------------------------------------------------------------------------- #
# parametric model
# --------------------------------------------------------------------------- #
def test_parametric_fit_and_sampling(reference_events):
    pm = ParametricModel().fit(reference_events)
    rng = np.random.default_rng(0)

    fd = pm.sample_fixation_duration(500, rng)
    assert fd.shape == (500,)
    assert np.all(np.isfinite(fd))
    assert np.all(fd > 0)
    assert 50 < np.mean(fd) < 1500

    sd = pm.sample_saccade_duration(500, rng)
    assert np.all(np.isfinite(sd))

    amp = pm.sample_saccade_amplitude(500, rng)
    assert np.all(np.isfinite(amp))
    assert np.all(amp >= 0)

    amp_e = pm.sample_saccade_amplitude_empirical(200, rng)
    assert np.all(np.isfinite(amp_e))

    noise = pm.sample_fixation_noise(500, rng)
    assert np.all(np.isfinite(noise))

    mr = pm.sample_missing_rate(rng)
    assert 0.0 <= mr <= 1.0

    pupil = pm.sample_pupil_size(10, rng)
    assert np.all(np.isfinite(pupil))


def test_parametric_robust_to_small_sample():
    tiny = RuleBasedGenerator(
        Config(experiment=ExperimentConfig(n_subjects=1, n_stimuli=1,
                                           trials_per_subject=1, seed=3))
    ).generate_events()
    pm = ParametricModel().fit(tiny)
    rng = np.random.default_rng(0)
    assert np.all(np.isfinite(pm.sample_fixation_duration(50, rng)))
    assert 0.0 <= pm.sample_missing_rate(rng) <= 1.0


def test_parametric_roundtrip(reference_events):
    pm = ParametricModel().fit(reference_events)
    pm2 = ParametricModel.from_dict(pm.to_dict())
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    np.testing.assert_allclose(
        pm.sample_fixation_duration(100, rng1),
        pm2.sample_fixation_duration(100, rng2),
    )


# --------------------------------------------------------------------------- #
# markov model
# --------------------------------------------------------------------------- #
def test_parse_and_classify():
    assert parse_aoi("L2_W5") == (2, 5)
    assert parse_aoi(None) is None
    assert parse_aoi("bad") is None
    assert classify_transition((0, 0), (0, 1)) == TransitionType.NEXT.value
    assert classify_transition((0, 0), (0, 3)) == TransitionType.SKIP.value
    assert classify_transition((0, 5), (0, 4)) == TransitionType.REGRESSION_1.value
    assert classify_transition((0, 5), (0, 2)) == TransitionType.REGRESSION_2PLUS.value
    assert classify_transition((0, 5), (1, 0)) == TransitionType.LINE_BREAK.value


def test_markov_valid_probabilities(reference_events):
    mm = MarkovTransitionModel().fit(reference_events)
    tm = mm.transition_matrix
    assert tm.shape == (len(STATES), len(STATES))
    assert np.all(tm >= -1e-12)
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(mm.start_distribution >= -1e-12)
    assert mm.start_distribution.sum() == pytest.approx(1.0)
    # END is absorbing.
    end_i = STATES.index(TransitionType.END.value)
    assert tm[end_i, end_i] == pytest.approx(1.0)


def test_markov_sampling_and_roundtrip(reference_events):
    mm = MarkovTransitionModel().fit(reference_events)
    rng = np.random.default_rng(0)
    seq = mm.sample_sequence(rng)
    assert all(s in STATES for s in seq)
    assert TransitionType.END.value not in seq  # sequence excludes terminal

    mm2 = MarkovTransitionModel.from_dict(mm.to_dict())
    np.testing.assert_allclose(mm.transition_matrix, mm2.transition_matrix)
    np.testing.assert_allclose(mm.start_distribution, mm2.start_distribution)


# --------------------------------------------------------------------------- #
# generator: events
# --------------------------------------------------------------------------- #
def test_fit_required_before_generate():
    g = ModelBasedGenerator(gen_config())
    with pytest.raises(RuntimeError):
        g.generate_events()


def test_generate_events_schema(fitted_gen):
    events = fitted_gen.generate_events(n_subjects=4, n_trials=3)
    validate_events(events)
    assert len(events) > 0
    types = set(events["event_type"].unique())
    assert EventType.FIXATION.value in types
    assert EventType.SACCADE.value in types
    # aoi only on fixations
    non_fix = events[events["event_type"] != EventType.FIXATION.value]
    assert non_fix["aoi_id"].isna().all()
    fix = events[events["event_type"] == EventType.FIXATION.value]
    assert fix["aoi_id"].notna().all()
    assert (fix["duration_ms"] > 0).all()


def test_generate_events_times_monotonic(fitted_gen):
    events = fitted_gen.generate_events(n_subjects=4, n_trials=3)
    for _, g in events.groupby(["subject_id", "trial_id"], sort=False):
        g = g.sort_values("event_index")
        assert (g["end_time_ms"] >= g["start_time_ms"]).all()
        assert np.allclose(g["duration_ms"], g["end_time_ms"] - g["start_time_ms"])
        assert g["start_time_ms"].is_monotonic_increasing
        assert list(g["event_index"]) == list(range(len(g)))


def test_generate_events_deterministic(reference_events):
    g1 = ModelBasedGenerator(gen_config(seed=11))
    g1.fit(reference_events)
    e1 = g1.generate_events(n_subjects=4, n_trials=3)

    g2 = ModelBasedGenerator(gen_config(seed=11))
    g2.fit(reference_events)
    e2 = g2.generate_events(n_subjects=4, n_trials=3)
    pd.testing.assert_frame_equal(e1, e2)


def test_n_subjects_n_trials_respected(fitted_gen):
    events = fitted_gen.generate_events(n_subjects=2, n_trials=2)
    assert events["subject_id"].nunique() == 2
    per_subj = events.groupby("subject_id")["trial_id"].nunique()
    assert (per_subj == 2).all()


# --------------------------------------------------------------------------- #
# generator: raw gaze + aoi summary
# --------------------------------------------------------------------------- #
def test_raw_gaze_schema_and_spacing(fitted_gen):
    events = fitted_gen.generate_events(n_subjects=4, n_trials=3)
    raw = fitted_gen.generate_raw_gaze(events)
    validate_raw_gaze(raw)
    assert len(raw) > 0
    dt = 1000.0 / fitted_gen.config.sampling.frequency_hz
    for _, g in raw.groupby(["subject_id", "trial_id"], sort=False):
        ts = g["timestamp_ms"].to_numpy()
        assert np.all(np.diff(ts) >= -1e-6)
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if len(diffs):
            assert np.median(diffs) == pytest.approx(dt, rel=0.05)


def test_raw_gaze_requires_events_first():
    g = ModelBasedGenerator(gen_config())
    with pytest.raises(RuntimeError):
        g.generate_raw_gaze(pd.DataFrame())


def test_aoi_summary_schema_and_counts(fitted_gen):
    events = fitted_gen.generate_events(n_subjects=4, n_trials=3)
    summary = fitted_gen.generate_aoi_summary(events)
    validate_aoi_summary(summary)
    assert len(summary) > 0
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
    assert (
        summary["total_fixation_duration_ms"]
        >= summary["first_fixation_duration_ms"] - 1e-6
    ).all()
    assert (summary["time_to_first_fixation_ms"] >= -1e-6).all()


# --------------------------------------------------------------------------- #
# save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_identical(reference_events, tmp_path):
    g = ModelBasedGenerator(gen_config(seed=11))
    g.fit(reference_events)
    e1 = g.generate_events(n_subjects=4, n_trials=3)

    path = tmp_path / "model.pkl"
    g.save(path)

    g2 = ModelBasedGenerator.load(path, config=gen_config(seed=11))
    e2 = g2.generate_events(n_subjects=4, n_trials=3)
    pd.testing.assert_frame_equal(e1, e2)

    # raw + aoi also conform after reload.
    validate_raw_gaze(g2.generate_raw_gaze(e2))
    validate_aoi_summary(g2.generate_aoi_summary(e2))


def test_load_without_config_uses_pickled_config(reference_events, tmp_path):
    g = ModelBasedGenerator(gen_config(seed=11))
    g.fit(reference_events)
    path = tmp_path / "model.pkl"
    g.save(path)
    g2 = ModelBasedGenerator.load(path)  # no config supplied
    events = g2.generate_events()
    validate_events(events)


def test_save_unfitted_raises(tmp_path):
    g = ModelBasedGenerator(gen_config())
    with pytest.raises(RuntimeError):
        g.save(tmp_path / "x.pkl")


# --------------------------------------------------------------------------- #
# optional HMM (skipped when hmmlearn is unavailable)
# --------------------------------------------------------------------------- #
def test_hmm_optional(reference_events):
    pytest.importorskip("hmmlearn")
    from synthetic_eye_tracking.models.hmm import HMMModel

    hmm = HMMModel(seed=0).fit(reference_events)
    assert hmm.fitted
    X, states = hmm.sample(20, np.random.default_rng(0))
    assert X.shape[0] == 20


def test_hmm_graceful_without_hmmlearn(reference_events):
    from synthetic_eye_tracking.models import hmm as hmm_mod

    if hmm_mod.hmmlearn_available():
        pytest.skip("hmmlearn installed; graceful-failure path not exercised")
    g = ModelBasedGenerator(gen_config(), use_hmm=True)
    with pytest.raises((RuntimeError, ImportError)):
        g.fit(reference_events)
