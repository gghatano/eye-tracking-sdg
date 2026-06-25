"""Participant (subject) generation (spec section 6, Step 1).

Each subject gets a small vector of latent traits that modulate the scanpath:
reading speed, fixation noise, regression/skip tendencies, missing-data rate and
a pupil baseline. Traits are sampled from ``config.participant_distribution`` and
clipped to sensible bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .schema import PARTICIPANT_COLUMNS


def generate_participants(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Return a participants DataFrame with ``PARTICIPANT_COLUMNS``."""
    pd_cfg = config.participant_distribution
    n = config.experiment.n_subjects
    rows: list[dict] = []

    for i in range(n):
        reading_speed = float(
            np.clip(
                rng.normal(pd_cfg.reading_speed_mean, pd_cfg.reading_speed_sd),
                0.25,
                4.0,
            )
        )
        # fixation noise sd must be strictly > 0.
        fixation_noise_sd = float(
            max(
                rng.normal(pd_cfg.fixation_noise_px_mean, pd_cfg.fixation_noise_px_sd),
                0.5,
            )
        )
        regression_tendency = float(
            np.clip(
                rng.normal(
                    pd_cfg.regression_tendency_mean, pd_cfg.regression_tendency_sd
                ),
                0.0,
                1.0,
            )
        )
        # No skip distribution in config: derive from the global skip_prob with
        # per-subject Gaussian noise so subjects differ but stay near the prior.
        skip_tendency = float(
            np.clip(
                rng.normal(config.transition.skip_prob, 0.03),
                0.0,
                1.0,
            )
        )
        missing_rate = float(
            np.clip(
                rng.normal(pd_cfg.missing_rate_mean, pd_cfg.missing_rate_sd),
                0.0,
                1.0,
            )
        )
        pupil_baseline = float(
            max(
                rng.normal(pd_cfg.pupil_baseline_mean, pd_cfg.pupil_baseline_sd),
                0.1,
            )
        )
        rows.append(
            {
                "subject_id": f"S{i:03d}",
                "reading_speed_factor": reading_speed,
                "fixation_noise_sd": fixation_noise_sd,
                "regression_tendency": regression_tendency,
                "skip_tendency": skip_tendency,
                "missing_rate": missing_rate,
                "pupil_baseline": pupil_baseline,
            }
        )

    return pd.DataFrame(rows, columns=PARTICIPANT_COLUMNS)
