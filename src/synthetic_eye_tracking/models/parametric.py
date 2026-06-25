"""Parametric marginal distribution model (spec section 7.3).

This module fits a small collection of univariate marginal distributions to a
reference events DataFrame and exposes ``sample_*`` methods used by the
model-based generator. Every fit is robust to small / degenerate samples: when
scipy MLE is unreliable we fall back to method-of-moments and finally to a
fixed prior so generation never crashes.

Fitted quantities (section 7.3):
* fixation_duration  -- lognormal or gamma (chosen by AIC)
* saccade_duration   -- normal or lognormal (chosen by AIC)
* saccade_amplitude  -- lognormal with empirical-quantile fallback
* x / y fixation noise -- normal (sd of fixation point residuals around AOI)
* missing_rate       -- beta, estimated from the validity == 0 fraction
* pupil_size         -- normal (skipped gracefully if pupil is absent)

All randomness flows through an explicit ``numpy.random.Generator`` so sampling
is reproducible. The model serialises to / from a plain ``dict`` for pickling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..schema import EventType

# Sensible priors used when a column is missing or too small to fit.
_FIX_DUR_PRIOR = {"family": "lognormal", "mu": math.log(220.0), "sigma": 0.4,
                  "min": 60.0, "max": 1200.0}
_SAC_DUR_PRIOR = {"family": "normal", "mu": 35.0, "sigma": 12.0,
                  "min": 10.0, "max": 120.0}
_SAC_AMP_PRIOR = {"family": "lognormal", "mu": math.log(110.0), "sigma": 0.6,
                  "min": 1.0, "max": 2000.0, "quantiles": None}
_NOISE_PRIOR = {"sd": 8.0}
_MISSING_PRIOR = {"alpha": 1.0, "beta": 30.0, "mean": 0.03}
_PUPIL_PRIOR = {"present": False, "mu": 3.5, "sigma": 0.4}

_MIN_FIT = 8  # below this many samples we use moment / prior fallbacks


def _clean(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return v


def _fit_positive_family(values: np.ndarray, prior: dict) -> dict:
    """Fit a positive continuous variable as lognormal or gamma (pick by AIC)."""
    v = _clean(values)
    v = v[v > 0]
    lo = float(np.min(v)) if v.size else prior.get("min", 0.0)
    hi = float(np.max(v)) if v.size else prior.get("max", np.inf)
    if v.size < _MIN_FIT:
        out = dict(prior)
        if v.size:
            out["min"] = max(out.get("min", lo), lo * 0.5)
            out["max"] = min(out.get("max", hi), hi * 1.5) if hi < np.inf else out.get("max")
        return out

    candidates: list[tuple[float, dict]] = []

    # Lognormal via closed-form MLE on log(values).
    logs = np.log(v)
    mu = float(np.mean(logs))
    sigma = float(max(np.std(logs, ddof=0), 1e-6))
    ll_ln = float(np.sum(stats.lognorm.logpdf(v, s=sigma, scale=math.exp(mu))))
    aic_ln = 2 * 2 - 2 * ll_ln
    candidates.append((aic_ln, {"family": "lognormal", "mu": mu, "sigma": sigma}))

    # Gamma via method-of-moments (stable; avoids scipy fit edge cases).
    m = float(np.mean(v))
    var = float(max(np.var(v, ddof=0), 1e-9))
    theta = var / m
    k = m / theta
    if k > 0 and theta > 0:
        ll_g = float(np.sum(stats.gamma.logpdf(v, a=k, scale=theta)))
        aic_g = 2 * 2 - 2 * ll_g
        candidates.append((aic_g, {"family": "gamma", "k": k, "theta": theta}))

    best = min(candidates, key=lambda c: c[0])[1]
    best["min"] = max(lo, 1e-6)
    best["max"] = hi
    return best


def _fit_normal_or_lognormal(values: np.ndarray, prior: dict) -> dict:
    """Fit a variable as normal or lognormal (pick by AIC). Used for saccade dur."""
    v = _clean(values)
    lo = float(np.min(v)) if v.size else prior.get("min", -np.inf)
    hi = float(np.max(v)) if v.size else prior.get("max", np.inf)
    if v.size < _MIN_FIT:
        return dict(prior)

    candidates: list[tuple[float, dict]] = []

    mu = float(np.mean(v))
    sd = float(max(np.std(v, ddof=0), 1e-6))
    ll_n = float(np.sum(stats.norm.logpdf(v, loc=mu, scale=sd)))
    candidates.append((2 * 2 - 2 * ll_n,
                       {"family": "normal", "mu": mu, "sigma": sd}))

    vp = v[v > 0]
    if vp.size >= _MIN_FIT:
        logs = np.log(vp)
        lmu = float(np.mean(logs))
        lsig = float(max(np.std(logs, ddof=0), 1e-6))
        ll_ln = float(np.sum(stats.lognorm.logpdf(vp, s=lsig, scale=math.exp(lmu))))
        candidates.append((2 * 2 - 2 * ll_ln,
                           {"family": "lognormal", "mu": lmu, "sigma": lsig}))

    best = min(candidates, key=lambda c: c[0])[1]
    best["min"] = lo
    best["max"] = hi
    return best


def _fit_normal(values: np.ndarray, default_sd: float) -> dict:
    v = _clean(values)
    if v.size < _MIN_FIT:
        return {"mu": 0.0, "sd": default_sd}
    return {"mu": float(np.mean(v)), "sd": float(max(np.std(v, ddof=0), 1e-6))}


def _fit_beta(fraction: float, n: int) -> dict:
    """Fit a Beta to a single observed fraction with weak prior smoothing.

    Uses the observed fraction and sample size as a pseudo-count so the Beta
    concentrates as more data is available. Always returns valid alpha/beta>0.
    """
    frac = float(np.clip(fraction, 1e-4, 1 - 1e-4))
    concentration = float(max(n, 2))
    alpha = frac * concentration + 1.0
    beta = (1.0 - frac) * concentration + 1.0
    return {"alpha": alpha, "beta": beta, "mean": frac}


def _sample_family(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` values from a fitted family spec and clip to [min, max]."""
    fam = spec.get("family", "lognormal")
    if fam == "lognormal":
        out = rng.lognormal(mean=spec["mu"], sigma=max(spec["sigma"], 1e-9), size=n)
    elif fam == "gamma":
        out = rng.gamma(shape=max(spec["k"], 1e-6), scale=max(spec["theta"], 1e-9),
                        size=n)
    elif fam == "normal":
        out = rng.normal(loc=spec["mu"], scale=max(spec["sigma"], 1e-9), size=n)
    elif fam == "empirical":
        q = np.asarray(spec["quantiles"], dtype=float)
        u = rng.random(n)
        out = np.interp(u, np.linspace(0.0, 1.0, len(q)), q)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown family: {fam!r}")
    lo = spec.get("min", -np.inf)
    hi = spec.get("max", np.inf)
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return np.clip(out, lo, hi)


class ParametricModel:
    """Fitted marginal distributions for the model-based generator (7.3)."""

    def __init__(self) -> None:
        self.fixation_duration: dict = dict(_FIX_DUR_PRIOR)
        self.saccade_duration: dict = dict(_SAC_DUR_PRIOR)
        self.saccade_amplitude: dict = dict(_SAC_AMP_PRIOR)
        self.fixation_noise: dict = dict(_NOISE_PRIOR)
        self.missing_rate: dict = dict(_MISSING_PRIOR)
        self.pupil_size: dict = dict(_PUPIL_PRIOR)
        self.fitted: bool = False

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, events: pd.DataFrame) -> "ParametricModel":
        fix = events[events["event_type"] == EventType.FIXATION.value]
        sac = events[events["event_type"] == EventType.SACCADE.value]

        # Fixation duration.
        self.fixation_duration = _fit_positive_family(
            fix["duration_ms"].to_numpy(), _FIX_DUR_PRIOR
        )

        # Saccade duration.
        if len(sac):
            self.saccade_duration = _fit_normal_or_lognormal(
                sac["duration_ms"].to_numpy(), _SAC_DUR_PRIOR
            )
        else:
            self.saccade_duration = dict(_SAC_DUR_PRIOR)

        # Saccade amplitude with empirical fallback when the parametric fit is
        # poor (heavy tails). We always also store empirical quantiles.
        amp = _clean(sac["saccade_amp_px"].to_numpy()) if len(sac) else np.array([])
        amp = amp[amp > 0]
        amp_spec = _fit_positive_family(amp, _SAC_AMP_PRIOR)
        if amp.size >= _MIN_FIT:
            amp_spec["quantiles"] = list(
                np.quantile(amp, np.linspace(0.0, 1.0, 21))
            )
        self.saccade_amplitude = amp_spec

        # Fixation x/y noise: residual of fixation points around the per-AOI mean
        # (the spatial scatter the generator must reproduce around AOI centres).
        self.fixation_noise = self._fit_noise(fix)

        # Missing rate from validity==0 fraction (over fixations).
        if len(fix):
            valid = fix["validity"].to_numpy(dtype=float)
            frac = float(np.mean(valid == 0))
            self.missing_rate = _fit_beta(frac, len(fix))
        else:
            self.missing_rate = dict(_MISSING_PRIOR)

        # Pupil size (events frame normally lacks pupil -> skip gracefully).
        if "pupil_size" in events.columns:
            pv = _clean(events["pupil_size"].to_numpy())
            if pv.size >= _MIN_FIT:
                self.pupil_size = {"present": True, **_fit_normal(pv, 0.4)}
                self.pupil_size["mu"] = self.pupil_size.pop("mu", 3.5)
                self.pupil_size["sigma"] = self.pupil_size.pop("sd", 0.4)
            else:
                self.pupil_size = dict(_PUPIL_PRIOR)
        else:
            self.pupil_size = dict(_PUPIL_PRIOR)

        self.fitted = True
        return self

    @staticmethod
    def _fit_noise(fix: pd.DataFrame) -> dict:
        """Std of fixation x/y around their per-(trial, AOI) mean position."""
        if not len(fix) or "aoi_id" not in fix.columns:
            return dict(_NOISE_PRIOR)
        sub = fix.dropna(subset=["aoi_id", "x_px", "y_px"]).copy()
        if len(sub) < _MIN_FIT:
            return dict(_NOISE_PRIOR)
        keys = ["subject_id", "trial_id", "aoi_id"]
        gx = sub.groupby(keys)["x_px"].transform("mean")
        gy = sub.groupby(keys)["y_px"].transform("mean")
        rx = (sub["x_px"] - gx).to_numpy()
        ry = (sub["y_px"] - gy).to_numpy()
        resid = np.concatenate([rx[np.isfinite(rx)], ry[np.isfinite(ry)]])
        if resid.size < _MIN_FIT:
            return dict(_NOISE_PRIOR)
        sd = float(max(np.std(resid, ddof=0), 0.5))
        return {"sd": sd}

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample_fixation_duration(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _sample_family(self.fixation_duration, n, rng)

    def sample_saccade_duration(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _sample_family(self.saccade_duration, n, rng)

    def sample_saccade_amplitude(self, n: int, rng: np.random.Generator) -> np.ndarray:
        spec = self.saccade_amplitude
        # Prefer the parametric family; the empirical quantiles are a fallback
        # available via sample_saccade_amplitude_empirical().
        return _sample_family(spec, n, rng)

    def sample_saccade_amplitude_empirical(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        q = self.saccade_amplitude.get("quantiles")
        if not q:
            return self.sample_saccade_amplitude(n, rng)
        spec = {"family": "empirical", "quantiles": q,
                "min": self.saccade_amplitude.get("min", 0.0),
                "max": self.saccade_amplitude.get("max", np.inf)}
        return _sample_family(spec, n, rng)

    def sample_fixation_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, max(self.fixation_noise["sd"], 1e-6), size=n)

    def sample_missing_rate(self, rng: np.random.Generator) -> float:
        a = max(self.missing_rate["alpha"], 1e-6)
        b = max(self.missing_rate["beta"], 1e-6)
        return float(rng.beta(a, b))

    def sample_pupil_size(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.pupil_size.get("mu", 3.5)
        sigma = self.pupil_size.get("sigma", 0.4)
        return rng.normal(mu, max(sigma, 1e-6), size=n)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "fixation_duration": self.fixation_duration,
            "saccade_duration": self.saccade_duration,
            "saccade_amplitude": self.saccade_amplitude,
            "fixation_noise": self.fixation_noise,
            "missing_rate": self.missing_rate,
            "pupil_size": self.pupil_size,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParametricModel":
        m = cls()
        m.fixation_duration = dict(data["fixation_duration"])
        m.saccade_duration = dict(data["saccade_duration"])
        m.saccade_amplitude = dict(data["saccade_amplitude"])
        m.fixation_noise = dict(data["fixation_noise"])
        m.missing_rate = dict(data["missing_rate"])
        m.pupil_size = dict(data["pupil_size"])
        m.fitted = bool(data.get("fitted", True))
        return m
