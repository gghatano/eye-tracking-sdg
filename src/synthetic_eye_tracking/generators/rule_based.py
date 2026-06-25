"""Rule-based synthetic eye-tracking generator (spec section 6).

The generator produces events, raw gaze and AOI summaries from a small set of
interpretable rules:

1. sample participants and stimuli (latent traits + difficulty),
2. for each subject x trial, pick a stimulus and build its AOI grid,
3. walk a scanpath over AOIs using transition probabilities modulated by the
   subject's regression/skip tendencies,
4. emit one fixation per visited AOI (point = AOI centre + noise, duration from
   the configured distribution scaled by difficulty and reading speed),
5. insert a saccade between consecutive fixations,
6. optionally insert a blink, and mark some fixations as missing.

Everything is driven by ``numpy.random.default_rng`` seeded from the config so
re-running with the same config yields byte-identical DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aoi import build_reading_grid
from ..config import Config
from ..events import (
    sample_distribution,
    saccade_amplitude,
    saccade_angle_deg,
    saccade_velocity_px_s,
)
from ..participants import generate_participants
from ..raw_gaze import expand_to_raw_gaze
from ..schema import (
    AOI_SUMMARY_COLUMNS,
    EVENT_COLUMNS,
    EventType,
    validate_aoi_summary,
    validate_events,
    validate_raw_gaze,
)
from ..stimuli import generate_stimuli
from .base import BaseGenerator

# Extra (internal) columns carried alongside EVENT_COLUMNS so raw-gaze expansion
# can interpolate saccades. They are stripped from the public events frame.
_INTERNAL_COLS = ["sacc_x0", "sacc_y0"]


class RuleBasedGenerator(BaseGenerator):
    """Rule-based generator (see module docstring and spec section 6)."""

    def __init__(self, config: Config):
        super().__init__(config)
        seed = config.experiment.seed
        # Top-level streams for participants/stimuli. The scanpath uses a fresh,
        # per-(subject, trial) seeded stream for locality + reproducibility.
        self.participants = generate_participants(
            config, np.random.default_rng(seed + 1)
        )
        self.stimuli = generate_stimuli(config, np.random.default_rng(seed + 2))

        # Pre-build AOI layouts for each stimulus (deterministic per stimulus).
        self._aoi_by_stimulus: dict[str, pd.DataFrame] = {}
        layouts: list[pd.DataFrame] = []
        for st_i, srow in enumerate(self.stimuli.itertuples(index=False)):
            layout_rng = np.random.default_rng(seed + 100 + st_i)
            layout = build_reading_grid(
                srow.stimulus_id,
                config,
                layout_rng,
                base_difficulty=float(srow.difficulty),
            )
            self._aoi_by_stimulus[srow.stimulus_id] = layout
            layouts.append(layout)
        self.aoi_layout = (
            pd.concat(layouts, ignore_index=True) if layouts else build_reading_grid(
                "ST000", config, np.random.default_rng(seed)
            ).iloc[0:0]
        )

        # Populated by generate_events() so raw-gaze expansion can use saccade
        # endpoints without polluting the public schema.
        self._events_internal: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Scanpath construction
    # ------------------------------------------------------------------ #
    def _build_scanpath(
        self, layout: pd.DataFrame, subject: pd.Series, rng: np.random.Generator
    ) -> list[int]:
        """Return a list of AOI row-indices (into ``layout``) for one trial.

        Movement is left->right; at a line end we line-break to the next line.
        We probabilistically skip ahead or regress, modulated by the subject's
        skip / regression tendencies.
        """
        t = self.config.transition
        n_aoi = len(layout)
        words_per_line = int(self.config.stimulus.words_per_line)

        # Per-subject modulation of the base transition probabilities.
        skip_p = float(np.clip(t.skip_prob + (subject.skip_tendency - t.skip_prob), 0, 1))
        regr_p = float(
            np.clip(t.regression_prob * (subject.regression_tendency / max(t.regression_prob, 1e-6)), 0, 1)
        )

        path: list[int] = [0]
        idx = 0
        # Cap path length to avoid pathological loops; reading is roughly linear.
        max_steps = n_aoi * 2
        steps = 0
        while idx < n_aoi - 1 and steps < max_steps:
            steps += 1
            line = idx // words_per_line
            word = idx % words_per_line
            at_line_end = word == words_per_line - 1

            r = rng.random()
            if at_line_end:
                # Either break to next line's first word, or (rarely) regress.
                if r < regr_p and len(path) > 1:
                    back = rng.integers(1, min(4, idx + 1))
                    idx = max(0, idx - int(back))
                else:
                    idx = (line + 1) * words_per_line  # first word of next line
                    if idx >= n_aoi:
                        break
            else:
                if r < regr_p and idx > 0:
                    back = rng.integers(1, min(4, idx + 1))
                    idx = max(0, idx - int(back))
                elif r < regr_p + skip_p and idx < n_aoi - 2:
                    idx = min(idx + 2, n_aoi - 1)
                else:
                    idx = idx + 1
            path.append(idx)
        return path

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def generate_events(self) -> pd.DataFrame:
        cfg = self.config
        seed = cfg.experiment.seed
        n_trials = cfg.experiment.trials_per_subject
        stim_ids = list(self.stimuli["stimulus_id"])
        fix_spec = cfg.event_distribution.fixation_duration_ms
        sac_spec = cfg.event_distribution.saccade_duration_ms

        rows: list[dict] = []

        for s_i, subj in enumerate(self.participants.itertuples(index=False)):
            for trial in range(n_trials):
                # Deterministic, locally-unique stream per (subject, trial).
                trial_seed = seed + 1_000_000 + s_i * 10_000 + trial
                rng = np.random.default_rng(trial_seed)

                stim_id = stim_ids[int(rng.integers(0, len(stim_ids)))]
                layout = self._aoi_by_stimulus[stim_id]
                trial_id = f"{subj.subject_id}_T{trial:03d}"

                path = self._build_scanpath(layout, subj, rng)

                event_index = 0
                t_now = 0.0
                prev_point: tuple[float, float] | None = None

                # Decide blink: insert before this fixation index (if any).
                blink_at = -1
                if rng.random() < cfg.blink.probability_per_trial and len(path) > 1:
                    blink_at = int(rng.integers(1, len(path)))

                for step, aoi_row_idx in enumerate(path):
                    aoi = layout.iloc[aoi_row_idx]
                    difficulty = float(aoi["difficulty"])

                    # Blink inserted just before this fixation.
                    if step == blink_at:
                        b_dur = float(
                            np.clip(
                                rng.normal(
                                    cfg.blink.duration_ms_mean, cfg.blink.duration_ms_sd
                                ),
                                30.0,
                                600.0,
                            )
                        )
                        rows.append(
                            self._make_event_row(
                                subj.subject_id,
                                trial_id,
                                stim_id,
                                event_index,
                                EventType.BLINK.value,
                                t_now,
                                t_now + b_dur,
                                x=prev_point[0] if prev_point else np.nan,
                                y=prev_point[1] if prev_point else np.nan,
                                aoi_id=None,
                                validity=0,
                            )
                        )
                        event_index += 1
                        t_now += b_dur

                    # Saccade from previous fixation to this AOI (skip for first).
                    cx = float(aoi["center_x"])
                    cy = float(aoi["center_y"])
                    noise_sd = float(subj.fixation_noise_sd) * (1.0 + 0.5 * difficulty)
                    fx = cx + rng.normal(0.0, noise_sd)
                    fy = cy + rng.normal(0.0, noise_sd)

                    if prev_point is not None:
                        sac_dur = float(sample_distribution(sac_spec, 1, rng)[0])
                        amp = saccade_amplitude(prev_point, (fx, fy))
                        ang = saccade_angle_deg(prev_point, (fx, fy))
                        vel = saccade_velocity_px_s(amp, sac_dur)
                        rows.append(
                            self._make_event_row(
                                subj.subject_id,
                                trial_id,
                                stim_id,
                                event_index,
                                EventType.SACCADE.value,
                                t_now,
                                t_now + sac_dur,
                                x=fx,
                                y=fy,
                                aoi_id=None,
                                validity=1,
                                saccade_amp_px=amp,
                                saccade_angle_deg=ang,
                                saccade_velocity_px_s=vel,
                                sacc_x0=prev_point[0],
                                sacc_y0=prev_point[1],
                            )
                        )
                        event_index += 1
                        t_now += sac_dur

                    # Fixation duration: base scaled by difficulty and (inverse)
                    # reading speed (faster reader -> shorter fixations).
                    base_dur = float(sample_distribution(fix_spec, 1, rng)[0])
                    scale = (1.0 + 0.6 * difficulty) / max(
                        float(subj.reading_speed_factor), 0.25
                    )
                    fix_dur = float(
                        np.clip(
                            base_dur * scale,
                            fix_spec.min if fix_spec.min is not None else base_dur,
                            fix_spec.max if fix_spec.max is not None else base_dur,
                        )
                    )

                    # Missing-data: mark this fixation invalid with prob missing_rate.
                    valid = 0 if rng.random() < float(subj.missing_rate) else 1

                    rows.append(
                        self._make_event_row(
                            subj.subject_id,
                            trial_id,
                            stim_id,
                            event_index,
                            EventType.FIXATION.value,
                            t_now,
                            t_now + fix_dur,
                            x=fx,
                            y=fy,
                            aoi_id=str(aoi["aoi_id"]),
                            validity=valid,
                        )
                    )
                    event_index += 1
                    t_now += fix_dur
                    prev_point = (fx, fy)

        internal = pd.DataFrame(rows, columns=EVENT_COLUMNS + _INTERNAL_COLS)
        self._events_internal = internal
        return validate_events(internal.drop(columns=_INTERNAL_COLS).copy())

    @staticmethod
    def _make_event_row(
        subject_id: str,
        trial_id: str,
        stimulus_id: str,
        event_index: int,
        event_type: str,
        start_time_ms: float,
        end_time_ms: float,
        *,
        x: float,
        y: float,
        aoi_id: str | None,
        validity: int,
        saccade_amp_px: float = np.nan,
        saccade_angle_deg: float = np.nan,
        saccade_velocity_px_s: float = np.nan,
        sacc_x0: float = np.nan,
        sacc_y0: float = np.nan,
    ) -> dict:
        return {
            "subject_id": subject_id,
            "trial_id": trial_id,
            "stimulus_id": stimulus_id,
            "event_index": int(event_index),
            "event_type": event_type,
            "start_time_ms": float(start_time_ms),
            "end_time_ms": float(end_time_ms),
            "duration_ms": float(end_time_ms - start_time_ms),
            "x_px": float(x) if x is not None and not _isnan(x) else np.nan,
            "y_px": float(y) if y is not None and not _isnan(y) else np.nan,
            "aoi_id": aoi_id,
            "saccade_amp_px": float(saccade_amp_px),
            "saccade_angle_deg": float(saccade_angle_deg),
            "saccade_velocity_px_s": float(saccade_velocity_px_s),
            "validity": int(validity),
            "sacc_x0": float(sacc_x0),
            "sacc_y0": float(sacc_y0),
        }

    # ------------------------------------------------------------------ #
    # Raw gaze
    # ------------------------------------------------------------------ #
    def generate_raw_gaze(self, events: pd.DataFrame) -> pd.DataFrame:
        # Use the internal frame (with saccade endpoints) when it matches the
        # supplied public events; otherwise reconstruct endpoints from the public
        # frame is not possible, so we require generate_events() to have run.
        if self._events_internal is None:
            raise RuntimeError("call generate_events() before generate_raw_gaze()")
        rng = np.random.default_rng(self.config.experiment.seed + 3)
        raw = expand_to_raw_gaze(
            self._events_internal, self.participants, self.config, rng
        )
        return validate_raw_gaze(raw)

    # ------------------------------------------------------------------ #
    # AOI summary
    # ------------------------------------------------------------------ #
    def generate_aoi_summary(self, events: pd.DataFrame) -> pd.DataFrame:
        fix = events[events["event_type"] == EventType.FIXATION.value].copy()
        fix = fix[fix["aoi_id"].notna()]
        if fix.empty:
            return validate_aoi_summary(
                pd.DataFrame({c: pd.Series(dtype="object") for c in AOI_SUMMARY_COLUMNS})
            )

        rows: list[dict] = []
        for (subject_id, trial_id, stimulus_id), g in fix.groupby(
            ["subject_id", "trial_id", "stimulus_id"], sort=False
        ):
            g = g.sort_values("event_index")
            trial_start = float(g["start_time_ms"].min())
            for aoi_id, ag in g.groupby("aoi_id", sort=False):
                ag = ag.sort_values("start_time_ms")
                durations = ag["duration_ms"].astype(float)
                fixation_count = int(len(ag))
                first_ffd = float(durations.iloc[0])
                total = float(durations.sum())
                ttff = float(ag["start_time_ms"].min()) - trial_start

                # Visit count = number of contiguous runs of this AOI in time.
                order = g.reset_index(drop=True)
                is_this = (order["aoi_id"] == aoi_id).to_numpy()
                visit_count = int(np.sum(is_this & ~np.r_[False, is_this[:-1]]))

                # Regression into this AOI: a fixation here that follows (in event
                # order) a fixation on a later AOI (i.e. a backward eye movement).
                regression_count = _count_regressions(order, aoi_id)

                rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "stimulus_id": stimulus_id,
                        "aoi_id": aoi_id,
                        "first_fixation_duration_ms": first_ffd,
                        "total_fixation_duration_ms": total,
                        "fixation_count": fixation_count,
                        "visit_count": max(visit_count, 1),
                        "time_to_first_fixation_ms": ttff,
                        "regression_count": regression_count,
                    }
                )

        df = pd.DataFrame(rows, columns=AOI_SUMMARY_COLUMNS)
        return validate_aoi_summary(df)


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _count_regressions(order: pd.DataFrame, aoi_id: str) -> int:
    """Count entries into ``aoi_id`` that arrive from a spatially-later AOI.

    We approximate AOI ordinal position by parsing ``L{line}_W{word}`` so a move
    from a higher ordinal to ``aoi_id`` counts as a regression.
    """

    def ordinal(a: str) -> int:
        try:
            line_s, word_s = a.split("_")
            return int(line_s[1:]) * 10_000 + int(word_s[1:])
        except Exception:
            return -1

    target = ordinal(aoi_id)
    aois = order["aoi_id"].tolist()
    count = 0
    for i in range(1, len(aois)):
        if aois[i] == aoi_id and ordinal(aois[i - 1]) > target:
            count += 1
    return count
