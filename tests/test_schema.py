"""Tests for the shared schema and config contract."""

from __future__ import annotations

import pandas as pd
import pytest

from synthetic_eye_tracking import schema
from synthetic_eye_tracking.config import Config, load_config


def test_config_defaults_match_spec():
    c = Config()
    assert c.experiment.seed == 42
    assert c.experiment.n_subjects == 100
    assert c.sampling.frequency_hz == 120
    assert c.stimulus.layout_type == "reading_grid"
    assert c.transition.forward_prob == 0.78
    # transition probabilities are a sensible distribution
    t = c.transition
    assert 0.99 <= t.forward_prob + t.skip_prob + t.regression_prob + t.line_break_prob <= 1.01


def test_config_roundtrip(tmp_path):
    c = Config()
    p = tmp_path / "cfg.yaml"
    c.to_yaml(p)
    c2 = Config.from_yaml(p)
    assert c2.model_dump() == c.model_dump()


def test_load_default_config_file():
    c = load_config("configs/default.yaml")
    assert c.experiment.n_subjects == 100
    assert c.event_distribution.fixation_duration_ms.distribution == "lognormal"


def test_load_config_none_returns_defaults():
    assert load_config(None).model_dump() == Config().model_dump()


def test_column_lists_are_unique_and_nonempty():
    for cols in (
        schema.EVENT_COLUMNS,
        schema.RAW_GAZE_COLUMNS,
        schema.AOI_SUMMARY_COLUMNS,
        schema.PARTICIPANT_COLUMNS,
        schema.STIMULUS_COLUMNS,
        schema.AOI_LAYOUT_COLUMNS,
    ):
        assert len(cols) == len(set(cols))
        assert len(cols) > 0


def test_event_type_enum_values():
    assert {e.value for e in schema.EventType} == {"fixation", "saccade", "blink"}


def test_ensure_columns_reorders():
    df = pd.DataFrame({c: [0] for c in reversed(schema.EVENT_COLUMNS)})
    out = schema.validate_events(df)
    assert list(out.columns) == schema.EVENT_COLUMNS


def test_ensure_columns_rejects_mismatch():
    df = pd.DataFrame({"subject_id": [1]})
    with pytest.raises(ValueError):
        schema.validate_events(df)


def test_empty_frame_has_columns():
    df = schema.empty_frame(schema.RAW_GAZE_COLUMNS)
    assert list(df.columns) == schema.RAW_GAZE_COLUMNS
    assert len(df) == 0
