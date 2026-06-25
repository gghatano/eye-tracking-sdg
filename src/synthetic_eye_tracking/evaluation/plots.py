"""Plotting helpers for the evaluation report (spec section 8).

Uses the non-interactive Agg backend so plots render without a display. Every
function is defensive: empty/small inputs are skipped rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..schema import EventType  # noqa: E402
from .metrics import (  # noqa: E402
    _clean_numeric,
    _extract_quantity,
    transition_matrix,
)

_QUANTITY_LABELS = {
    "fixation_duration_ms": "Fixation duration (ms)",
    "saccade_duration_ms": "Saccade duration (ms)",
    "saccade_amp_px": "Saccade amplitude (px)",
    "saccade_velocity_px_s": "Saccade velocity (px/s)",
    "scanpath_length": "Scanpath length (px)",
    "trial_duration": "Trial duration (ms)",
    "missing_rate": "Missing rate",
}


def _savefig(fig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_distribution(
    ref_df: pd.DataFrame, syn_df: pd.DataFrame, quantity: str, path: str | Path
) -> Path | None:
    """Overlaid reference/synthetic histograms for ``quantity``.

    Returns the written path, or None if there is nothing to plot.
    """
    ref = _extract_quantity(ref_df, quantity)
    syn = _extract_quantity(syn_df, quantity)
    if ref.size == 0 and syn.size == 0:
        return None

    label = _QUANTITY_LABELS.get(quantity, quantity)
    combined = np.concatenate([a for a in (ref, syn) if a.size])
    nbins = max(5, min(40, int(np.sqrt(combined.size)) if combined.size else 5))
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, nbins + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    if ref.size:
        ax.hist(ref, bins=bins, alpha=0.55, label="reference", density=True,
                color="#1f77b4")
    if syn.size:
        ax.hist(syn, bins=bins, alpha=0.55, label="synthetic", density=True,
                color="#ff7f0e")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_title(f"Distribution: {label}")
    ax.legend()
    return _savefig(fig, path)


def plot_transition_matrix(events_df: pd.DataFrame, path: str | Path) -> Path | None:
    """Heatmap of the AOI->AOI transition probability matrix."""
    mat, labels = transition_matrix(events_df)
    if not labels:
        return None
    # Cap displayed labels to keep the heatmap legible.
    max_labels = 30
    if len(labels) > max_labels:
        labels = labels[:max_labels]
        mat = mat.loc[labels, labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat.to_numpy(dtype="float64"), cmap="viridis", aspect="auto",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("to AOI")
    ax.set_ylabel("from AOI")
    ax.set_title("AOI transition probabilities")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _savefig(fig, path)


def plot_scanpath_examples(
    events_df: pd.DataFrame, path: str | Path, n: int = 4
) -> Path | None:
    """Plot up to ``n`` example scanpaths (connected fixation x/y points)."""
    if events_df is None or len(events_df) == 0:
        return None
    fix = events_df[events_df["event_type"] == EventType.FIXATION.value].copy()
    if len(fix) == 0:
        return None
    fix["event_index"] = pd.to_numeric(fix["event_index"], errors="coerce")
    fix["x_px"] = pd.to_numeric(fix["x_px"], errors="coerce")
    fix["y_px"] = pd.to_numeric(fix["y_px"], errors="coerce")

    groups = list(fix.groupby(["subject_id", "trial_id"], dropna=False))
    groups = [g for g in groups if len(g[1]) >= 2][:n]
    if not groups:
        return None

    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, (key, grp) in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        grp = grp.sort_values("event_index")
        x = _clean_numeric(grp["x_px"])
        y = _clean_numeric(grp["y_px"])
        m = min(x.size, y.size)
        x, y = x[:m], y[:m]
        if m >= 1:
            ax.plot(x, y, "-o", color="#2c7fb8", markersize=4, linewidth=1)
            if m >= 1:
                ax.scatter(x[0], y[0], color="green", s=40, zorder=3, label="start")
                ax.scatter(x[-1], y[-1], color="red", s=40, zorder=3, label="end")
        ax.invert_yaxis()  # screen coordinates: y grows downward
        ax.set_title(f"subj={key[0]} trial={key[1]}", fontsize=8)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    # Hide any unused axes.
    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Example scanpaths")
    return _savefig(fig, path)


def generate_all_plots(
    ref_df: pd.DataFrame, syn_df: pd.DataFrame, out_dir: str | Path
) -> dict[str, Path]:
    """Generate the standard plot set into ``out_dir``.

    Always attempts the four canonical plots; entries are omitted when their
    underlying data is empty. Never raises on small/empty data.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    specs = [
        ("fixation_duration_distribution",
         lambda p: plot_distribution(ref_df, syn_df, "fixation_duration_ms", p)),
        ("saccade_amplitude_distribution",
         lambda p: plot_distribution(ref_df, syn_df, "saccade_amp_px", p)),
        ("transition_matrix", lambda p: plot_transition_matrix(syn_df, p)),
        ("scanpath_examples", lambda p: plot_scanpath_examples(syn_df, p)),
    ]
    for name, fn in specs:
        path = out / f"{name}.png"
        try:
            result = fn(path)
        except Exception:
            result = None
        if result is not None:
            written[name] = result
    return written
