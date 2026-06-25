"""Synthetic-data adapter for the sim-to-real event-detection experiment (M3-A).

Two responsibilities:

1. :func:`rawgaze_to_samples` -- convert the rule-based generator's raw-gaze
   output (``RAW_GAZE_COLUMNS``) into the common labeled sample frame
   (``SAMPLE_COLUMNS``), converting pixels to degrees of visual angle and
   mapping ``event_type`` to the detector's label space.

2. :func:`generate_synthetic_samples` -- produce a **domain-randomized**
   synthetic training set. Each domain draws a randomized screen geometry and
   perturbs the generator's configuration, so a detector trained on the union
   sees a spread of setups and transfers better to a real recording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    BlinkConfig,
    Config,
    DistributionSpec,
    EventDistributionConfig,
    ExperimentConfig,
    ParticipantDistributionConfig,
    SamplingConfig,
    ScreenConfig,
    TransitionConfig,
)
from ..generators.rule_based import RuleBasedGenerator
from ..schema import EventType
from . import (
    BLINK,
    FIXATION,
    OTHER,
    SACCADE,
    pix_to_deg,
    validate_samples,
)

# Map the generator's per-sample event_type to the detector's label space.
_EVENT_TYPE_TO_LABEL: dict[str, str] = {
    EventType.FIXATION.value: FIXATION,
    EventType.SACCADE.value: SACCADE,
    EventType.BLINK.value: BLINK,
}


def rawgaze_to_samples(
    raw_gaze_df: pd.DataFrame,
    *,
    screen_res_px: tuple[float, float],
    screen_size_m: tuple[float, float],
    view_dist_m: float,
) -> pd.DataFrame:
    """Convert a raw-gaze frame into a validated labeled sample frame.

    ``series_id`` is ``f"syn_{subject_id}_{trial_id}"`` and ``source`` is
    ``"synthetic"``. Pixel coordinates are converted to degrees via
    :func:`pix_to_deg`; missing samples (``validity == 0`` or NaN ``x_px``)
    stay NaN. ``event_type`` maps to FIXATION / SACCADE / BLINK, with any other
    value -> OTHER.
    """
    raw = raw_gaze_df

    x_px = raw["x_px"].to_numpy(dtype=float)
    y_px = raw["y_px"].to_numpy(dtype=float)
    validity = raw["validity"].to_numpy()

    x_deg, y_deg = pix_to_deg(
        x_px,
        y_px,
        screen_res_px=screen_res_px,
        screen_size_m=screen_size_m,
        view_dist_m=view_dist_m,
    )

    # Missing whenever validity is 0 or the pixel position is NaN.
    missing = (validity == 0) | ~np.isfinite(x_px) | ~np.isfinite(y_px)
    x_deg = np.where(missing, np.nan, x_deg)
    y_deg = np.where(missing, np.nan, y_deg)

    labels = (
        raw["event_type"].map(lambda e: _EVENT_TYPE_TO_LABEL.get(e, OTHER)).to_numpy()
    )

    series_id = (
        "syn_"
        + raw["subject_id"].astype(str)
        + "_"
        + raw["trial_id"].astype(str)
    ).to_numpy()

    samples = pd.DataFrame(
        {
            "series_id": series_id,
            "source": "synthetic",
            "t_ms": raw["timestamp_ms"].to_numpy(dtype=float),
            "x_deg": x_deg,
            "y_deg": y_deg,
            "label": labels,
        }
    )
    return validate_samples(samples, name="synthetic_samples")


def _domain_config(
    rng: np.random.Generator,
    *,
    domain_seed: int,
    frequency_hz: float,
    n_subjects: int,
    trials_per_subject: int,
    n_stimuli: int,
) -> Config:
    """Build a Config for one randomized domain from the base defaults."""
    base = Config()
    pd_base = base.participant_distribution
    ev_base = base.event_distribution
    tr_base = base.transition

    # Perturb fixation positional noise.
    noise_sd = float(pd_base.fixation_noise_px_sd * rng.uniform(0.7, 1.5))

    # Jitter transition probabilities slightly, then keep them in [0, 1].
    def jit(p: float) -> float:
        return float(np.clip(p * rng.uniform(0.85, 1.15), 0.0, 1.0))

    transition = TransitionConfig(
        forward_prob=jit(tr_base.forward_prob),
        skip_prob=jit(tr_base.skip_prob),
        regression_prob=jit(tr_base.regression_prob),
        line_break_prob=jit(tr_base.line_break_prob),
    )

    # Modestly vary event duration means about their defaults.
    fix_spec = ev_base.fixation_duration_ms
    sac_spec = ev_base.saccade_duration_ms
    event_distribution = EventDistributionConfig(
        fixation_duration_ms=DistributionSpec(
            distribution=fix_spec.distribution,
            mean=float(fix_spec.mean * rng.uniform(0.9, 1.1)),
            sd=fix_spec.sd,
            min=fix_spec.min,
            max=fix_spec.max,
        ),
        saccade_duration_ms=DistributionSpec(
            distribution=sac_spec.distribution,
            mean=float(sac_spec.mean * rng.uniform(0.9, 1.1)),
            sd=sac_spec.sd,
            min=sac_spec.min,
            max=sac_spec.max,
        ),
    )

    participant_distribution = ParticipantDistributionConfig(
        **{
            **pd_base.model_dump(),
            "fixation_noise_px_sd": noise_sd,
        }
    )

    return Config(
        experiment=ExperimentConfig(
            seed=domain_seed,
            n_subjects=n_subjects,
            n_stimuli=n_stimuli,
            trials_per_subject=trials_per_subject,
        ),
        screen=ScreenConfig(),  # 1920x1080 default
        sampling=SamplingConfig(frequency_hz=frequency_hz),
        participant_distribution=participant_distribution,
        event_distribution=event_distribution,
        transition=transition,
        blink=BlinkConfig(),
    )


def generate_synthetic_samples(
    *,
    n_domains: int = 6,
    seed: int = 0,
    frequency_hz: float = 500.0,
    n_subjects: int = 4,
    trials_per_subject: int = 3,
    n_stimuli: int = 2,
) -> pd.DataFrame:
    """Generate a domain-randomized synthetic labeled sample frame.

    For each of ``n_domains`` domains a seeded RNG draws a randomized screen
    geometry (physical size, viewing distance) and perturbs the generator
    config (fixation noise, transition probabilities, event-duration means,
    sampling frequency, and a distinct seed). The rule-based generator is run,
    its raw gaze converted with that domain's geometry, and ``series_id`` is
    prefixed per domain so ids are unique across domains. All domains are
    concatenated and returned as one ``validate_samples``-valid frame.
    """
    master = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    for d in range(n_domains):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))

        # Randomized screen geometry.
        width_m = float(rng.uniform(0.30, 0.60))
        height_m = width_m * 0.6
        view_dist_m = float(rng.uniform(0.55, 0.75))

        cfg = _domain_config(
            rng,
            domain_seed=seed + 1000 + d,
            frequency_hz=frequency_hz,
            n_subjects=n_subjects,
            trials_per_subject=trials_per_subject,
            n_stimuli=n_stimuli,
        )
        screen_res_px = (float(cfg.screen.width_px), float(cfg.screen.height_px))

        gen = RuleBasedGenerator(cfg)
        events = gen.generate_events()
        raw = gen.generate_raw_gaze(events)

        samples = rawgaze_to_samples(
            raw,
            screen_res_px=screen_res_px,
            screen_size_m=(width_m, height_m),
            view_dist_m=view_dist_m,
        )
        # Prefix series_id with the domain so ids stay unique across domains.
        samples = samples.copy()
        samples["series_id"] = f"d{d}_" + samples["series_id"].astype(str)
        frames.append(samples)

    combined = pd.concat(frames, ignore_index=True)
    return validate_samples(combined, name="synthetic_samples")
