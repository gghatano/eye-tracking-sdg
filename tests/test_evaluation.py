"""Tests for the evaluation framework (spec section 8)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from synthetic_eye_tracking.schema import EVENT_COLUMNS, EventType, validate_events
from synthetic_eye_tracking.evaluation import metrics, plots, report


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_events(
    n_subjects: int = 3,
    n_trials: int = 2,
    n_fix: int = 6,
    seed: int = 0,
    fix_mean: float = 240.0,
    amp_mean: float = 90.0,
) -> pd.DataFrame:
    """Fabricate a valid EVENT_COLUMNS DataFrame with fixations + saccades.

    Each trial alternates fixation/saccade events. AOIs follow ``L{line}_W{word}``
    advancing word-by-word (with the occasional line break) so transition and
    sequence parsing has something to chew on.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for s in range(n_subjects):
        for t in range(n_trials):
            clock = 0.0
            event_index = 0
            line, word = 0, 0
            for k in range(n_fix):
                fix_dur = float(max(80.0, rng.normal(fix_mean, 40.0)))
                x = 200.0 + word * 110.0 + rng.normal(0, 5)
                y = 180.0 + line * 60.0
                rows.append({
                    "subject_id": f"S{s:02d}",
                    "trial_id": t,
                    "stimulus_id": f"stim{t}",
                    "event_index": event_index,
                    "event_type": EventType.FIXATION.value,
                    "start_time_ms": clock,
                    "end_time_ms": clock + fix_dur,
                    "duration_ms": fix_dur,
                    "x_px": x,
                    "y_px": y,
                    "aoi_id": f"L{line}_W{word}",
                    "saccade_amp_px": np.nan,
                    "saccade_angle_deg": np.nan,
                    "saccade_velocity_px_s": np.nan,
                    "validity": 0 if rng.random() < 0.05 else 1,
                })
                clock += fix_dur
                event_index += 1

                if k < n_fix - 1:
                    sacc_dur = float(max(15.0, rng.normal(35.0, 8.0)))
                    amp = float(max(5.0, rng.normal(amp_mean, 20.0)))
                    vel = amp / (sacc_dur / 1000.0)
                    rows.append({
                        "subject_id": f"S{s:02d}",
                        "trial_id": t,
                        "stimulus_id": f"stim{t}",
                        "event_index": event_index,
                        "event_type": EventType.SACCADE.value,
                        "start_time_ms": clock,
                        "end_time_ms": clock + sacc_dur,
                        "duration_ms": sacc_dur,
                        "x_px": np.nan,
                        "y_px": np.nan,
                        "aoi_id": None,
                        "saccade_amp_px": amp,
                        "saccade_angle_deg": float(rng.uniform(-180, 180)),
                        "saccade_velocity_px_s": vel,
                        "validity": 1,
                    })
                    clock += sacc_dur
                    event_index += 1

                # Advance AOI position (occasional line break).
                word += 1
                if word >= 5:
                    word = 0
                    line += 1
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return validate_events(df)


@pytest.fixture
def ref_df():
    return make_events(seed=1)


@pytest.fixture
def syn_df():
    # Slightly shifted so distances are non-zero but finite.
    return make_events(seed=2, fix_mean=255.0, amp_mean=85.0)


# --------------------------------------------------------------------------- #
# distribution_metrics
# --------------------------------------------------------------------------- #


def test_distribution_metrics_keys_and_finite(ref_df, syn_df):
    out = metrics.distribution_metrics(ref_df, syn_df)
    assert set(out) == set(metrics.DISTRIBUTION_QUANTITIES)
    for q, m in out.items():
        for src in ("reference", "synthetic"):
            for stat in ("mean", "std", "median", "iqr", "n"):
                assert stat in m[src]
        assert math.isfinite(m["ks_statistic"]), q
        assert math.isfinite(m["wasserstein"]), q
        assert 0.0 <= m["ks_statistic"] <= 1.0


def test_distribution_metrics_empty_no_crash():
    empty = make_events().iloc[0:0]
    out = metrics.distribution_metrics(empty, empty)
    for q, m in out.items():
        assert m["reference"]["n"] == 0
        assert math.isnan(m["ks_statistic"])


# --------------------------------------------------------------------------- #
# transition matrix
# --------------------------------------------------------------------------- #


def test_transition_matrix_square_rows_sum_one(syn_df):
    mat, labels = metrics.transition_matrix(syn_df)
    assert mat.shape[0] == mat.shape[1] == len(labels)
    row_sums = mat.sum(axis=1)
    for a in labels:
        s = row_sums[a]
        # Rows with outgoing transitions sum to ~1; terminal AOIs sum to 0.
        assert abs(s - 1.0) < 1e-9 or abs(s) < 1e-9


def test_transition_matrix_distance_self_zero(syn_df):
    assert metrics.transition_matrix_distance(syn_df, syn_df) == pytest.approx(0.0)


def test_transition_matrix_empty():
    empty = make_events().iloc[0:0]
    mat, labels = metrics.transition_matrix(empty)
    assert labels == []
    assert math.isnan(metrics.transition_matrix_distance(empty, empty))


# --------------------------------------------------------------------------- #
# sequence + privacy
# --------------------------------------------------------------------------- #


def test_sequence_metrics(ref_df, syn_df):
    out = metrics.sequence_metrics(ref_df, syn_df)
    assert "bigram_jaccard" in out
    assert "reference" in out and "synthetic" in out
    for key in ("regression_rate", "skip_rate", "line_transition_rate"):
        assert key in out["reference"]


def test_privacy_metrics(ref_df, syn_df):
    out = metrics.privacy_metrics(ref_df, syn_df)
    assert out["n_synthetic_records"] > 0
    assert math.isfinite(out["mean_nn_distance"])
    assert 0.0 <= out["reidentification_rate"] <= 1.0


def test_privacy_metrics_empty():
    empty = make_events().iloc[0:0]
    out = metrics.privacy_metrics(empty, empty)
    assert math.isnan(out["mean_nn_distance"])


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def test_generate_all_plots(ref_df, syn_df, tmp_path):
    paths = plots.generate_all_plots(ref_df, syn_df, tmp_path)
    for name in ("fixation_duration_distribution", "saccade_amplitude_distribution",
                 "transition_matrix", "scanpath_examples"):
        assert name in paths
        assert paths[name].exists()


def test_generate_all_plots_empty_no_crash(tmp_path):
    empty = make_events().iloc[0:0]
    paths = plots.generate_all_plots(empty, empty, tmp_path)
    assert isinstance(paths, dict)  # may be empty, must not crash


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_build_report_writes_outputs(ref_df, syn_df, tmp_path):
    out = metrics_dict = report.build_report(ref_df, syn_df, tmp_path)
    assert (tmp_path / "quality_report.md").exists()
    assert (tmp_path / "quality_report.json").exists()
    plots_dir = tmp_path / "plots"
    assert plots_dir.is_dir()
    for name in ("fixation_duration_distribution", "saccade_amplitude_distribution",
                 "transition_matrix", "scanpath_examples"):
        assert (plots_dir / f"{name}.png").exists()

    md = (tmp_path / "quality_report.md").read_text()
    for section in ("Basic statistics comparison", "Distribution distance metrics",
                    "AOI transition comparison", "Scanpath examples",
                    "Privacy proxy metrics",
                    "Recommendation on whether synthetic data is usable"):
        assert section in md
    assert "verdict" in metrics_dict
    assert metrics_dict["self_comparison"] is False
    _ = out


def test_build_report_self_comparison(syn_df, tmp_path):
    metrics_dict = report.build_report(syn_df, syn_df, tmp_path, label="self")
    assert metrics_dict["self_comparison"] is True
    assert (tmp_path / "quality_report.md").exists()
    assert metrics_dict["transition_matrix_distance"] == pytest.approx(0.0)
