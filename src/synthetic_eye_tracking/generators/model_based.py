"""Model-based synthetic eye-tracking generator (spec section 7).

Where the rule-based generator hard-codes interpretable rules, this generator
*learns* its behaviour from a reference events DataFrame:

* a :class:`ParametricModel` captures the marginal distributions of fixation
  duration, saccade duration / amplitude, spatial fixation noise, the missing
  fraction and (optionally) pupil size (section 7.3);
* a :class:`MarkovTransitionModel` captures the relative AOI-transition dynamics
  over the :class:`TransitionType` state space (section 7.4);
* an optional :class:`HMMModel` (section 7.5) is supported but OFF by default and
  never required.

Generation walks a concrete reading grid by sampling relative transition states
from the Markov chain, placing one fixation per visited AOI at its centre plus
sampled noise with a sampled duration, and inserting a saccade between
consecutive fixations. Output conforms exactly to ``schema.EVENT_COLUMNS`` and
is fully reproducible from ``config.experiment.seed``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..aoi import build_reading_grid
from ..config import Config
from ..events import saccade_angle_deg, saccade_velocity_px_s
from ..models.hmm import HMMModel
from ..models.markov import MarkovTransitionModel
from ..models.parametric import ParametricModel
from ..raw_gaze import expand_to_raw_gaze
from ..schema import (
    AOI_SUMMARY_COLUMNS,
    EVENT_COLUMNS,
    PARTICIPANT_COLUMNS,
    EventType,
    TransitionType,
    validate_aoi_summary,
    validate_events,
    validate_raw_gaze,
)
from .base import BaseGenerator

# Internal columns carried alongside EVENT_COLUMNS for raw-gaze saccade
# interpolation (stripped from the public events frame), matching rule_based.
_INTERNAL_COLS = ["sacc_x0", "sacc_y0"]


class ModelBasedGenerator(BaseGenerator):
    """Generator driven by fitted parametric + Markov models (spec 7)."""

    def __init__(self, config: Config, *, use_hmm: bool = False) -> None:
        super().__init__(config)
        self.parametric = ParametricModel()
        self.markov = MarkovTransitionModel()
        self.use_hmm = bool(use_hmm)
        self.hmm: HMMModel | None = None
        self._fitted = False
        # Populated by generate_events() so raw-gaze expansion can use saccade
        # endpoints without polluting the public schema.
        self._events_internal: pd.DataFrame | None = None
        self._participants: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, events: pd.DataFrame) -> "ModelBasedGenerator":
        """Fit the parametric + Markov models (and optional HMM) to reference."""
        self.parametric.fit(events)
        self.markov.fit(events)
        if self.use_hmm:
            self.hmm = HMMModel(seed=self.config.experiment.seed)
            self.hmm.fit(events)  # raises clearly if hmmlearn missing
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # AOI layout
    # ------------------------------------------------------------------ #
    def _stimulus_id(self, index: int) -> str:
        return f"ST{index:03d}"

    def _build_layouts(self) -> dict[str, pd.DataFrame]:
        """Deterministic per-stimulus reading grids (same scheme as rule_based)."""
        cfg = self.config
        seed = cfg.experiment.seed
        layouts: dict[str, pd.DataFrame] = {}
        rng_diff = np.random.default_rng(seed + 2)
        for st_i in range(cfg.experiment.n_stimuli):
            stim_id = self._stimulus_id(st_i)
            base_difficulty = float(np.clip(rng_diff.uniform(0.2, 0.8), 0.0, 1.0))
            layout = build_reading_grid(
                stim_id,
                cfg,
                np.random.default_rng(seed + 100 + st_i),
                base_difficulty=base_difficulty,
            )
            layouts[stim_id] = layout
        return layouts

    # ------------------------------------------------------------------ #
    # Scanpath: translate relative transition states into AOI row indices
    # ------------------------------------------------------------------ #
    def _walk(
        self, states: list[str], words_per_line: int, n_aoi: int
    ) -> list[int]:
        """Translate a sampled state sequence into AOI row indices on the grid."""
        path: list[int] = [0]
        idx = 0
        for state in states:
            line = idx // words_per_line
            word = idx % words_per_line
            if state == TransitionType.NEXT.value:
                nxt = idx + 1
            elif state == TransitionType.SKIP.value:
                nxt = idx + 2
            elif state == TransitionType.REGRESSION_1.value:
                nxt = idx - 1
            elif state == TransitionType.REGRESSION_2PLUS.value:
                nxt = idx - 2
            elif state == TransitionType.LINE_BREAK.value:
                nxt = (line + 1) * words_per_line  # first word of next line
            else:  # END or unknown
                break

            if nxt < 0:
                nxt = 0
            if nxt >= n_aoi:
                # Past the end: if we were moving forward, stop; otherwise clamp.
                if state in (
                    TransitionType.NEXT.value,
                    TransitionType.SKIP.value,
                    TransitionType.LINE_BREAK.value,
                ):
                    break
                nxt = n_aoi - 1
            path.append(int(nxt))
            idx = int(nxt)
        return path

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def generate_events(
        self, n_subjects: int | None = None, n_trials: int | None = None
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() before generate_events()")

        cfg = self.config
        seed = cfg.experiment.seed
        n_subjects = int(n_subjects if n_subjects is not None
                         else cfg.experiment.n_subjects)
        n_trials = int(n_trials if n_trials is not None
                       else cfg.experiment.trials_per_subject)
        words_per_line = int(cfg.stimulus.words_per_line)

        layouts = self._build_layouts()
        stim_ids = list(layouts.keys())

        rows: list[dict] = []
        participant_rows: list[dict] = []

        for s_i in range(n_subjects):
            subject_id = f"S{s_i:03d}"
            # One subject-level missing rate (and pupil baseline) sampled once.
            subj_rng = np.random.default_rng(seed + 500_000 + s_i)
            missing_rate = self.parametric.sample_missing_rate(subj_rng)
            pupil_baseline = float(self.parametric.sample_pupil_size(1, subj_rng)[0])
            participant_rows.append(
                {
                    "subject_id": subject_id,
                    "reading_speed_factor": 1.0,
                    "fixation_noise_sd": float(
                        max(self.parametric.fixation_noise["sd"], 0.5)
                    ),
                    "regression_tendency": 0.0,
                    "skip_tendency": 0.0,
                    "missing_rate": missing_rate,
                    "pupil_baseline": pupil_baseline,
                }
            )

            for trial in range(n_trials):
                trial_seed = seed + 1_000_000 + s_i * 10_000 + trial
                rng = np.random.default_rng(trial_seed)

                stim_id = stim_ids[int(rng.integers(0, len(stim_ids)))]
                layout = layouts[stim_id]
                n_aoi = len(layout)
                trial_id = f"{subject_id}_T{trial:03d}"

                states = self.markov.sample_sequence(rng)
                path = self._walk(states, words_per_line, n_aoi)

                event_index = 0
                t_now = 0.0
                prev_point: tuple[float, float] | None = None

                # Pre-sample per-fixation quantities in one vectorised draw.
                n_fix = len(path)
                durs = self.parametric.sample_fixation_duration(n_fix, rng)
                noise = self.parametric.sample_fixation_noise(2 * n_fix, rng)

                # Optional blink before a random fixation, reusing the missing
                # mechanism's spirit: probability tied to the fitted missing rate.
                blink_at = -1
                if n_fix > 1 and rng.random() < min(0.5, 3.0 * missing_rate + 0.05):
                    blink_at = int(rng.integers(1, n_fix))

                for step, aoi_row_idx in enumerate(path):
                    aoi = layout.iloc[aoi_row_idx]
                    cx = float(aoi["center_x"])
                    cy = float(aoi["center_y"])
                    fx = cx + float(noise[2 * step])
                    fy = cy + float(noise[2 * step + 1])

                    # Blink inserted just before this fixation.
                    if step == blink_at:
                        b_dur = float(
                            np.clip(self.parametric.sample_saccade_duration(1, rng)[0]
                                    * 3.0, 40.0, 600.0)
                        )
                        rows.append(
                            self._make_event_row(
                                subject_id, trial_id, stim_id, event_index,
                                EventType.BLINK.value, t_now, t_now + b_dur,
                                x=prev_point[0] if prev_point else np.nan,
                                y=prev_point[1] if prev_point else np.nan,
                                aoi_id=None, validity=0,
                            )
                        )
                        event_index += 1
                        t_now += b_dur

                    # Saccade from previous fixation to this AOI.
                    if prev_point is not None:
                        sac_dur = float(
                            self.parametric.sample_saccade_duration(1, rng)[0]
                        )
                        sac_dur = float(np.clip(sac_dur, 5.0, 300.0))
                        amp = float(
                            np.hypot(fx - prev_point[0], fy - prev_point[1])
                        )
                        ang = saccade_angle_deg(prev_point, (fx, fy))
                        vel = saccade_velocity_px_s(amp, sac_dur)
                        rows.append(
                            self._make_event_row(
                                subject_id, trial_id, stim_id, event_index,
                                EventType.SACCADE.value, t_now, t_now + sac_dur,
                                x=fx, y=fy, aoi_id=None, validity=1,
                                saccade_amp_px=amp, saccade_angle_deg=ang,
                                saccade_velocity_px_s=vel,
                                sacc_x0=prev_point[0], sacc_y0=prev_point[1],
                            )
                        )
                        event_index += 1
                        t_now += sac_dur

                    fix_dur = float(durs[step])
                    valid = 0 if rng.random() < missing_rate else 1
                    rows.append(
                        self._make_event_row(
                            subject_id, trial_id, stim_id, event_index,
                            EventType.FIXATION.value, t_now, t_now + fix_dur,
                            x=fx, y=fy, aoi_id=str(aoi["aoi_id"]), validity=valid,
                        )
                    )
                    event_index += 1
                    t_now += fix_dur
                    prev_point = (fx, fy)

        internal = pd.DataFrame(rows, columns=EVENT_COLUMNS + _INTERNAL_COLS)
        self._events_internal = internal
        self._participants = pd.DataFrame(
            participant_rows, columns=PARTICIPANT_COLUMNS
        )
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
        if self._events_internal is None or self._participants is None:
            raise RuntimeError("call generate_events() before generate_raw_gaze()")
        rng = np.random.default_rng(self.config.experiment.seed + 3)
        raw = expand_to_raw_gaze(
            self._events_internal, self._participants, self.config, rng
        )
        return validate_raw_gaze(raw)

    # ------------------------------------------------------------------ #
    # AOI summary (reimplements rule_based aggregation; same columns)
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
            order = g.reset_index(drop=True)
            for aoi_id, ag in g.groupby("aoi_id", sort=False):
                ag = ag.sort_values("start_time_ms")
                durations = ag["duration_ms"].astype(float)
                fixation_count = int(len(ag))
                first_ffd = float(durations.iloc[0])
                total = float(durations.sum())
                ttff = float(ag["start_time_ms"].min()) - trial_start

                is_this = (order["aoi_id"] == aoi_id).to_numpy()
                visit_count = int(np.sum(is_this & ~np.r_[False, is_this[:-1]]))
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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("cannot save an unfitted model; call fit() first")
        payload = {
            "version": 1,
            "config": self.config.model_dump(),
            "parametric": self.parametric.to_dict(),
            "markov": self.markov.to_dict(),
            "use_hmm": self.use_hmm,
            "hmm": self.hmm.to_dict() if self.hmm is not None else None,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(
        cls, path: str | Path, config: Config | None = None
    ) -> "ModelBasedGenerator":
        with open(Path(path), "rb") as fh:
            payload = pickle.load(fh)
        cfg = config if config is not None else Config.from_dict(payload["config"])
        gen = cls(cfg, use_hmm=bool(payload.get("use_hmm", False)))
        gen.parametric = ParametricModel.from_dict(payload["parametric"])
        gen.markov = MarkovTransitionModel.from_dict(payload["markov"])
        if payload.get("hmm") is not None:
            gen.hmm = HMMModel.from_dict(payload["hmm"])
        gen._fitted = True
        return gen


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _count_regressions(order: pd.DataFrame, aoi_id: str) -> int:
    """Count entries into ``aoi_id`` arriving from a spatially-later AOI."""

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
