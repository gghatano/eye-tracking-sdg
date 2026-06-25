"""Stimulus generation (spec section 6, Step 2).

Each stimulus is a ``reading_grid`` of a fixed geometry (from the config) with a
scalar ``difficulty`` in [0, 1] sampled per stimulus. The difficulty seeds the
per-AOI difficulties built by :mod:`synthetic_eye_tracking.aoi`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .schema import STIMULUS_COLUMNS


def generate_stimuli(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Return a stimuli DataFrame with ``STIMULUS_COLUMNS``."""
    s = config.stimulus
    n = config.experiment.n_stimuli
    n_aoi = s.n_lines * s.words_per_line
    rows: list[dict] = []

    for i in range(n):
        difficulty = float(np.clip(rng.uniform(0.2, 0.8), 0.0, 1.0))
        rows.append(
            {
                "stimulus_id": f"ST{i:03d}",
                "layout_type": s.layout_type,
                "n_lines": int(s.n_lines),
                "words_per_line": int(s.words_per_line),
                "n_aoi": int(n_aoi),
                "difficulty": difficulty,
            }
        )

    return pd.DataFrame(rows, columns=STIMULUS_COLUMNS)
