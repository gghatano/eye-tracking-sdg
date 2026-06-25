"""Optional Gaussian HMM over reading states (spec section 7.5).

This is an *optional* model. The default model-based generator does NOT require
it. ``hmmlearn`` ships in the ``[hmm]`` extra; if it is not installed the class
still imports and ``fit`` raises a clear, catchable error with an install hint.

When available, the HMM models the observation sequence
``[fixation_duration, saccade_amplitude]`` per trial with a few hidden states
loosely interpreted as reading regimes (normal / careful / skimming / rereading).
It is primarily a diagnostic / extension hook; the generator falls back to the
parametric + Markov path when the HMM is unavailable or not fitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..schema import EventType

STATE_NAMES = ["normal_reading", "careful_reading", "skimming", "rereading"]

_INSTALL_HINT = (
    "hmmlearn is not installed. Install the optional extra to use HMMModel:\n"
    "    uv pip install 'synthetic-eye-tracking[hmm]'\n"
    "or:  uv pip install hmmlearn"
)


def hmmlearn_available() -> bool:
    try:
        import hmmlearn  # noqa: F401

        return True
    except Exception:
        return False


class HMMModel:
    """Thin wrapper around a Gaussian HMM (optional; section 7.5)."""

    def __init__(self, n_states: int = 4, n_iter: int = 50, seed: int = 0) -> None:
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self._model = None
        self.fitted = False

    # ------------------------------------------------------------------ #
    # Observation extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sequences(events: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Build [fixation_duration, next_saccade_amplitude] obs per trial.

        Returns the stacked observation array and the per-trial lengths expected
        by ``hmmlearn``'s ``fit(X, lengths)`` API.
        """
        obs: list[list[float]] = []
        lengths: list[int] = []
        for _, g in events.groupby(["subject_id", "trial_id"], sort=False):
            g = g.sort_values("event_index")
            fix = g[g["event_type"] == EventType.FIXATION.value]
            if fix.empty:
                continue
            durs = fix["duration_ms"].to_numpy(dtype=float)
            sac = g[g["event_type"] == EventType.SACCADE.value]
            amps = sac["saccade_amp_px"].to_numpy(dtype=float)
            n = len(durs)
            for i in range(n):
                amp = float(amps[i]) if i < len(amps) and np.isfinite(amps[i]) else 0.0
                obs.append([float(durs[i]), amp])
            lengths.append(n)
        if not obs:
            return np.empty((0, 2), dtype=float), []
        return np.asarray(obs, dtype=float), lengths

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, events: pd.DataFrame) -> "HMMModel":
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # ImportError or backend issues
            raise RuntimeError(_INSTALL_HINT) from exc

        X, lengths = self._sequences(events)
        if X.shape[0] < self.n_states * 2 or not lengths:
            raise RuntimeError(
                "not enough observation data to fit the HMM "
                f"(have {X.shape[0]} observations, need >= {self.n_states * 2})"
            )

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        model.fit(X, lengths)
        self._model = model
        self.fitted = True
        return self

    def sample(self, n: int, rng: np.random.Generator | None = None):
        if not self.fitted or self._model is None:
            raise RuntimeError("HMMModel is not fitted; call fit() first.")
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        X, states = self._model.sample(n, random_state=seed)
        return X, states

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "seed": self.seed,
            "fitted": self.fitted,
        }
        if self.fitted and self._model is not None:
            data["model"] = {
                "startprob": self._model.startprob_.tolist(),
                "transmat": self._model.transmat_.tolist(),
                "means": self._model.means_.tolist(),
                "covars": self._model.covars_.tolist(),
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HMMModel":
        m = cls(
            n_states=int(data.get("n_states", 4)),
            n_iter=int(data.get("n_iter", 50)),
            seed=int(data.get("seed", 0)),
        )
        model_data = data.get("model")
        if model_data:
            try:
                from hmmlearn.hmm import GaussianHMM
            except Exception:
                return m  # can't rebuild without hmmlearn; leave unfitted
            gm = GaussianHMM(n_components=m.n_states, covariance_type="diag")
            gm.startprob_ = np.asarray(model_data["startprob"], dtype=float)
            gm.transmat_ = np.asarray(model_data["transmat"], dtype=float)
            gm.means_ = np.asarray(model_data["means"], dtype=float)
            covars = np.asarray(model_data["covars"], dtype=float)
            # GaussianHMM stores diagonal covars internally as (n, n_features).
            gm._covars_ = covars if covars.ndim == 2 else np.array(
                [np.diag(c) for c in covars]
            )
            m._model = gm
            m.fitted = True
        return m
