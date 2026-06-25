"""AOI layout generation for ``reading_grid`` stimuli (spec section 6, Step 2).

An AOI layout places a grid of rectangular areas-of-interest on the screen,
laid out left->right within a line and top->bottom across lines. Each AOI gets
a per-AOI ``difficulty`` in [0, 1] sampled deterministically from the rng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .schema import AOI_LAYOUT_COLUMNS


def build_reading_grid(
    stimulus_id: str,
    config: Config,
    rng: np.random.Generator,
    *,
    base_difficulty: float = 0.5,
) -> pd.DataFrame:
    """Build the AOI layout for one stimulus.

    AOIs are laid out from ``margin_left_px`` / ``margin_top_px`` using the
    stimulus geometry. ``aoi_id`` is ``L{line}_W{word}``. The per-AOI difficulty
    is centred on ``base_difficulty`` (the stimulus-level difficulty) with small
    Gaussian jitter and clipped to [0, 1].
    """
    s = config.stimulus
    rows: list[dict] = []

    for line in range(s.n_lines):
        y_min = s.margin_top_px + line * (s.aoi_height_px + s.line_spacing_px)
        y_max = y_min + s.aoi_height_px
        center_y = y_min + s.aoi_height_px / 2.0
        for word in range(s.words_per_line):
            x_min = s.margin_left_px + word * s.aoi_width_px
            x_max = x_min + s.aoi_width_px
            center_x = x_min + s.aoi_width_px / 2.0
            difficulty = float(
                np.clip(base_difficulty + rng.normal(0.0, 0.12), 0.0, 1.0)
            )
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "aoi_id": f"L{line}_W{word}",
                    "line_index": line,
                    "word_index": word,
                    "x_min": float(x_min),
                    "x_max": float(x_max),
                    "y_min": float(y_min),
                    "y_max": float(y_max),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "difficulty": difficulty,
                }
            )

    df = pd.DataFrame(rows, columns=AOI_LAYOUT_COLUMNS)
    return df
