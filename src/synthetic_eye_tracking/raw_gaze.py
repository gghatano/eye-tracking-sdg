"""Expand event-level rows into a raw gaze time-series (spec section 6, Step 6).

Events describe fixations / saccades / blinks with start and end times. For each
trial we lay a single continuous time grid at ``config.sampling.frequency_hz``
(so timestamps are strictly increasing and evenly spaced) and assign every grid
sample to the event that covers it:

* fixation -> sample jitters around the fixation point (Gaussian noise),
* saccade  -> sample linearly interpolates between the saccade endpoints,
* blink    -> x / y / pupil_size are NaN and validity = 0.

Pupil size hovers around the participant baseline with small noise (NaN whenever
the sample is invalid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .schema import RAW_GAZE_COLUMNS, EventType


def expand_to_raw_gaze(
    events: pd.DataFrame,
    participants: pd.DataFrame,
    config: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Expand ``events`` into a raw gaze DataFrame with ``RAW_GAZE_COLUMNS``."""
    if events.empty:
        return pd.DataFrame(
            {c: pd.Series(dtype="object") for c in RAW_GAZE_COLUMNS}
        )

    freq = float(config.sampling.frequency_hz)
    dt_ms = 1000.0 / freq

    pupil_by_subject = dict(
        zip(participants["subject_id"], participants["pupil_baseline"])
    )

    cols: dict[str, list] = {c: [] for c in RAW_GAZE_COLUMNS}

    # Process one trial at a time on a single continuous grid so timestamps are
    # strictly monotonic and evenly spaced.
    for (subject_id, trial_id, stimulus_id), trial in events.groupby(
        ["subject_id", "trial_id", "stimulus_id"], sort=False
    ):
        trial = trial.sort_values("start_time_ms")
        t_start = float(trial["start_time_ms"].min())
        t_end = float(trial["end_time_ms"].max())
        if t_end <= t_start:
            continue

        n = int(np.floor((t_end - t_start) / dt_ms)) + 1
        grid = t_start + np.arange(n) * dt_ms
        baseline = float(pupil_by_subject.get(subject_id, 3.5))

        # Map each grid sample to its covering event (last event whose start <= t).
        starts = trial["start_time_ms"].to_numpy(dtype=float)
        ev_idx_for_sample = np.searchsorted(starts, grid, side="right") - 1
        ev_idx_for_sample = np.clip(ev_idx_for_sample, 0, len(trial) - 1)

        rows = list(trial.itertuples(index=False))

        for gi in range(n):
            ev = rows[ev_idx_for_sample[gi]]
            t = grid[gi]
            etype = ev.event_type

            if etype == EventType.BLINK.value:
                x = y = pupil = np.nan
                validity = 0
            elif etype == EventType.SACCADE.value:
                span = max(float(ev.end_time_ms) - float(ev.start_time_ms), 1e-6)
                frac = np.clip((t - float(ev.start_time_ms)) / span, 0.0, 1.0)
                x0, y0 = float(ev.sacc_x0), float(ev.sacc_y0)
                x1, y1 = float(ev.x_px), float(ev.y_px)
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                pupil = baseline + float(rng.normal(0.0, 0.05))
                validity = int(ev.validity)
            else:  # fixation
                if int(ev.validity) == 0:
                    x = y = pupil = np.nan
                    validity = 0
                else:
                    x = float(ev.x_px) + float(rng.normal(0.0, 2.0))
                    y = float(ev.y_px) + float(rng.normal(0.0, 2.0))
                    pupil = baseline + float(rng.normal(0.0, 0.05))
                    validity = 1

            aoi_val = ev.aoi_id if isinstance(ev.aoi_id, str) else None

            cols["subject_id"].append(subject_id)
            cols["trial_id"].append(trial_id)
            cols["stimulus_id"].append(stimulus_id)
            cols["timestamp_ms"].append(float(t))
            cols["x_px"].append(x)
            cols["y_px"].append(y)
            cols["pupil_size"].append(pupil)
            cols["validity"].append(int(validity))
            cols["event_index"].append(int(ev.event_index))
            cols["event_type"].append(etype)
            cols["aoi_id"].append(aoi_val)

    return pd.DataFrame(cols, columns=RAW_GAZE_COLUMNS)
