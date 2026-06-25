"""Markov AOI-transition model (spec section 7.4).

Reading is modelled as a walk over AOIs. Rather than learning transitions
between absolute AOIs (which would not generalise across stimuli of different
sizes) we classify each consecutive within-trial fixation transition into the
relative :class:`~synthetic_eye_tracking.schema.TransitionType` state space:

* ``next``              -- advance one word on the same line
* ``skip``              -- advance two or more words on the same line
* ``regression_1``      -- move back exactly one word
* ``regression_2plus``  -- move back two or more words
* ``line_break``        -- move to a different (typically the next) line
* ``end``               -- terminal state (end of scanpath)

We learn a first-order Markov chain over these states (a transition matrix plus
a start distribution). At generation time we sample a sequence of relative moves
and translate them into AOI indices over a concrete reading grid.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..schema import EventType, TransitionType

# Canonical ordered state list (defines matrix row/column order).
STATES: list[str] = [
    TransitionType.NEXT.value,
    TransitionType.SKIP.value,
    TransitionType.REGRESSION_1.value,
    TransitionType.REGRESSION_2PLUS.value,
    TransitionType.LINE_BREAK.value,
    TransitionType.END.value,
]
_IDX = {s: i for i, s in enumerate(STATES)}
_N = len(STATES)

_AOI_RE = re.compile(r"^L(\d+)_W(\d+)$")

# Prior transition matrix (rows = from-state) used to smooth sparse fits. Rows
# sum to 1; END is absorbing. Encodes "mostly read forward".
_PRIOR_ROWS = {
    "next":             [0.74, 0.08, 0.06, 0.02, 0.07, 0.03],
    "skip":             [0.70, 0.08, 0.07, 0.03, 0.08, 0.04],
    "regression_1":     [0.68, 0.06, 0.08, 0.04, 0.08, 0.06],
    "regression_2plus": [0.66, 0.06, 0.08, 0.05, 0.09, 0.06],
    "line_break":       [0.78, 0.07, 0.04, 0.02, 0.04, 0.05],
    "end":              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
_PRIOR_START = [0.82, 0.06, 0.03, 0.01, 0.06, 0.02]


def parse_aoi(aoi_id: Any) -> tuple[int, int] | None:
    """Parse ``L{line}_W{word}`` into ``(line, word)`` or None if unparseable."""
    if not isinstance(aoi_id, str):
        return None
    m = _AOI_RE.match(aoi_id)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def classify_transition(prev: tuple[int, int], cur: tuple[int, int]) -> str:
    """Classify a relative AOI move into a :class:`TransitionType` value."""
    pl, pw = prev
    cl, cw = cur
    if cl != pl:
        return TransitionType.LINE_BREAK.value
    dw = cw - pw
    if dw >= 2:
        return TransitionType.SKIP.value
    if dw == 1:
        return TransitionType.NEXT.value
    if dw == -1:
        return TransitionType.REGRESSION_1.value
    if dw <= -2:
        return TransitionType.REGRESSION_2PLUS.value
    # dw == 0 (refixation): treat as a (degenerate) forward move.
    return TransitionType.NEXT.value


class MarkovTransitionModel:
    """First-order Markov chain over relative AOI transition states (7.4)."""

    def __init__(self) -> None:
        self.transition_matrix: np.ndarray = self._prior_matrix()
        self.start_distribution: np.ndarray = np.asarray(_PRIOR_START, dtype=float)
        self.mean_path_length: float = 12.0
        self.fitted: bool = False

    @staticmethod
    def _prior_matrix() -> np.ndarray:
        return np.asarray([_PRIOR_ROWS[s] for s in STATES], dtype=float)

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, events: pd.DataFrame, *, smoothing: float = 1.0) -> "MarkovTransitionModel":
        fix = events[events["event_type"] == EventType.FIXATION.value]
        fix = fix[fix["aoi_id"].notna()]

        # Laplace-smoothed counts seeded with the prior (so unseen rows are sane).
        counts = self._prior_matrix() * smoothing
        start_counts = np.asarray(_PRIOR_START, dtype=float) * smoothing
        path_lengths: list[int] = []

        for _, g in fix.groupby(["subject_id", "trial_id"], sort=False):
            g = g.sort_values("event_index")
            coords = [parse_aoi(a) for a in g["aoi_id"].tolist()]
            coords = [c for c in coords if c is not None]
            if len(coords) < 2:
                if coords:
                    path_lengths.append(len(coords))
                continue
            path_lengths.append(len(coords))

            states: list[str] = []
            for prev, cur in zip(coords[:-1], coords[1:]):
                states.append(classify_transition(prev, cur))

            if states:
                start_counts[_IDX[states[0]]] += 1.0
            for a, b in zip(states[:-1], states[1:]):
                counts[_IDX[a], _IDX[b]] += 1.0
            # Each scanpath ends -> transition from last observed state to END.
            if states:
                counts[_IDX[states[-1]], _IDX[TransitionType.END.value]] += 1.0

        # END is absorbing.
        counts[_IDX[TransitionType.END.value], :] = 0.0
        counts[_IDX[TransitionType.END.value], _IDX[TransitionType.END.value]] = 1.0

        self.transition_matrix = self._normalise_rows(counts)
        self.start_distribution = start_counts / start_counts.sum()
        if path_lengths:
            self.mean_path_length = float(np.mean(path_lengths))
        self.fitted = True
        return self

    @staticmethod
    def _normalise_rows(counts: np.ndarray) -> np.ndarray:
        rs = counts.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return counts / rs

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample_start_state(self, rng: np.random.Generator) -> str:
        i = int(rng.choice(_N, p=self.start_distribution))
        return STATES[i]

    def sample_next_state(self, current_state: str, rng: np.random.Generator) -> str:
        row = self.transition_matrix[_IDX[current_state]]
        i = int(rng.choice(_N, p=row))
        return STATES[i]

    def sample_sequence(
        self, rng: np.random.Generator, *, max_steps: int = 200
    ) -> list[str]:
        """Sample a sequence of (non-END) transition states for one scanpath."""
        seq: list[str] = []
        state = self.sample_start_state(rng)
        steps = 0
        while state != TransitionType.END.value and steps < max_steps:
            seq.append(state)
            state = self.sample_next_state(state, rng)
            steps += 1
        return seq

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "states": STATES,
            "transition_matrix": self.transition_matrix.tolist(),
            "start_distribution": self.start_distribution.tolist(),
            "mean_path_length": self.mean_path_length,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkovTransitionModel":
        m = cls()
        m.transition_matrix = np.asarray(data["transition_matrix"], dtype=float)
        m.start_distribution = np.asarray(data["start_distribution"], dtype=float)
        m.mean_path_length = float(data.get("mean_path_length", 12.0))
        m.fitted = bool(data.get("fitted", True))
        return m
