"""Low-level sampling helpers for event generation (spec section 6).

These functions are intentionally small and pure so the rule-based generator
(and tests) can compose them. Every stochastic helper takes an explicit
``numpy.random.Generator`` so output is fully reproducible.
"""

from __future__ import annotations

import math

import numpy as np

from .config import DistributionSpec


def sample_distribution(
    spec: DistributionSpec, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``size`` values from ``spec`` and truncate to ``[min, max]``.

    Supported families: ``lognormal``, ``normal``, ``gamma``. ``mean``/``sd`` are
    interpreted on the natural (data) scale; for the lognormal we solve for the
    underlying normal parameters so the requested mean/sd hold approximately.
    Out-of-bounds samples are clipped (not resampled) which keeps the call O(size)
    and deterministic.
    """
    if size <= 0:
        return np.empty(0, dtype=float)

    dist = spec.distribution.lower()
    mean = float(spec.mean)
    sd = float(spec.sd)

    if dist == "normal":
        out = rng.normal(loc=mean, scale=max(sd, 1e-9), size=size)
    elif dist == "lognormal":
        # Convert desired data-scale mean/sd to underlying normal parameters.
        mean = max(mean, 1e-9)
        var = sd * sd
        sigma2 = math.log(1.0 + var / (mean * mean))
        sigma = math.sqrt(max(sigma2, 1e-12))
        mu = math.log(mean) - 0.5 * sigma2
        out = rng.lognormal(mean=mu, sigma=sigma, size=size)
    elif dist == "gamma":
        sd = max(sd, 1e-9)
        mean = max(mean, 1e-9)
        # mean = k*theta, var = k*theta^2 -> theta = var/mean, k = mean/theta
        theta = (sd * sd) / mean
        k = mean / theta
        out = rng.gamma(shape=k, scale=theta, size=size)
    else:
        raise ValueError(f"unsupported distribution: {spec.distribution!r}")

    lo = spec.min if spec.min is not None else -np.inf
    hi = spec.max if spec.max is not None else np.inf
    return np.clip(out, lo, hi)


def saccade_amplitude(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Euclidean distance (px) between two gaze points."""
    return float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def saccade_angle_deg(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Direction of travel p0 -> p1 in degrees, atan2(dy, dx), range (-180, 180]."""
    return float(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))


def saccade_velocity_px_s(amplitude_px: float, duration_ms: float) -> float:
    """Mean velocity in px/s given amplitude and duration. Safe for tiny durations."""
    duration_s = max(duration_ms, 1e-6) / 1000.0
    return float(amplitude_px / duration_s)
