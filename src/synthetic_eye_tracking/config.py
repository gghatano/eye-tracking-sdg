"""Configuration models and YAML loading.

The configuration is the single source of truth for every generator and the
evaluation pipeline. It mirrors the YAML structure documented in the spec
(section 5) and is validated with pydantic so misconfiguration fails fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Base):
    seed: int = 42
    n_subjects: int = 100
    n_stimuli: int = 20
    trials_per_subject: int = 20


class ScreenConfig(_Base):
    width_px: int = 1920
    height_px: int = 1080


class SamplingConfig(_Base):
    frequency_hz: float = 120.0


class StimulusConfig(_Base):
    layout_type: str = "reading_grid"
    n_lines: int = 8
    words_per_line: int = 12
    aoi_width_px: float = 110.0
    aoi_height_px: float = 36.0
    line_spacing_px: float = 60.0
    margin_left_px: float = 200.0
    margin_top_px: float = 180.0


class ParticipantDistributionConfig(_Base):
    reading_speed_mean: float = 1.0
    reading_speed_sd: float = 0.2
    fixation_noise_px_mean: float = 0.0
    fixation_noise_px_sd: float = 8.0
    regression_tendency_mean: float = 0.12
    regression_tendency_sd: float = 0.05
    missing_rate_mean: float = 0.03
    missing_rate_sd: float = 0.02
    # Pupil baseline is referenced by the rule-based generator (Step 1).
    pupil_baseline_mean: float = 3.5
    pupil_baseline_sd: float = 0.4


class DistributionSpec(_Base):
    """A named univariate distribution with truncation bounds."""

    distribution: str = "lognormal"
    mean: float = 0.0
    sd: float = 1.0
    min: float | None = None
    max: float | None = None


class EventDistributionConfig(_Base):
    fixation_duration_ms: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec(
            distribution="lognormal", mean=240, sd=80, min=80, max=800
        )
    )
    saccade_duration_ms: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec(
            distribution="normal", mean=35, sd=10, min=15, max=80
        )
    )


class TransitionConfig(_Base):
    forward_prob: float = 0.78
    skip_prob: float = 0.08
    regression_prob: float = 0.10
    line_break_prob: float = 0.04


class BlinkConfig(_Base):
    probability_per_trial: float = 0.15
    duration_ms_mean: float = 120.0
    duration_ms_sd: float = 50.0


class Config(_Base):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    stimulus: StimulusConfig = Field(default_factory=StimulusConfig)
    participant_distribution: ParticipantDistributionConfig = Field(
        default_factory=ParticipantDistributionConfig
    )
    event_distribution: EventDistributionConfig = Field(
        default_factory=EventDistributionConfig
    )
    transition: TransitionConfig = Field(default_factory=TransitionConfig)
    blink: BlinkConfig = Field(default_factory=BlinkConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate a config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)
        )


def load_config(path: str | Path | None) -> Config:
    """Load a config from ``path`` or return defaults when ``path`` is None."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
