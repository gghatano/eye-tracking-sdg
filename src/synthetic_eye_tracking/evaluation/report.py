"""Assemble the quality report (spec sections 8 & 12 acceptance).

``build_report`` computes all metrics, renders plots and writes both a JSON and
a Markdown report. It supports self-comparison (reference == synthetic) which is
the M1 case where no separate reference corpus exists.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..io.writers import write_json, write_text
from .metrics import (
    DISTRIBUTION_QUANTITIES,
    distribution_metrics,
    privacy_metrics,
    sequence_metrics,
    transition_matrix_distance,
)
from .plots import generate_all_plots

# Heuristic thresholds for the usability verdict.
_KS_GOOD = 0.15
_KS_OK = 0.30
_TRANSITION_GOOD = 0.5
_TRANSITION_OK = 1.0


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "n/a"
    if math.isinf(xf):
        return "inf"
    return f"{xf:.{nd}f}"


def _mean_ignore_nan(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


def _verdict(dist: dict, transition_distance: float) -> dict[str, Any]:
    ks_values = [dist[q]["ks_statistic"] for q in dist]
    wass_values = [dist[q]["wasserstein"] for q in dist]
    agg_ks = _mean_ignore_nan(ks_values)
    agg_wass = _mean_ignore_nan(wass_values)

    reasons: list[str] = []
    score = "usable"

    if math.isnan(agg_ks):
        reasons.append("Could not compute aggregate KS statistic (insufficient data).")
        score = "inconclusive"
    elif agg_ks <= _KS_GOOD:
        reasons.append(f"Aggregate KS {agg_ks:.3f} is low (good distributional match).")
    elif agg_ks <= _KS_OK:
        reasons.append(f"Aggregate KS {agg_ks:.3f} is moderate.")
        if score == "usable":
            score = "usable_with_caution"
    else:
        reasons.append(f"Aggregate KS {agg_ks:.3f} is high (distributions diverge).")
        score = "not_usable"

    if not math.isnan(transition_distance):
        if transition_distance <= _TRANSITION_GOOD:
            reasons.append(
                f"Transition-matrix distance {transition_distance:.3f} is low (scanpath "
                "dynamics preserved)."
            )
        elif transition_distance <= _TRANSITION_OK:
            reasons.append(
                f"Transition-matrix distance {transition_distance:.3f} is moderate."
            )
            if score == "usable":
                score = "usable_with_caution"
        else:
            reasons.append(
                f"Transition-matrix distance {transition_distance:.3f} is high (scanpath "
                "dynamics differ)."
            )
            if score in ("usable", "usable_with_caution"):
                score = "not_usable"

    return {
        "verdict": score,
        "aggregate_ks": agg_ks,
        "aggregate_wasserstein": agg_wass,
        "transition_distance": transition_distance,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _md_basic_stats(dist: dict) -> str:
    lines = ["## Basic statistics comparison", ""]
    lines.append("| Quantity | Ref mean | Syn mean | Ref std | Syn std | Ref median | Syn median | Ref IQR | Syn IQR | n ref | n syn |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for q in DISTRIBUTION_QUANTITIES:
        r = dist[q]["reference"]
        s = dist[q]["synthetic"]
        lines.append(
            f"| {q} | {_fmt(r['mean'])} | {_fmt(s['mean'])} | {_fmt(r['std'])} | "
            f"{_fmt(s['std'])} | {_fmt(r['median'])} | {_fmt(s['median'])} | "
            f"{_fmt(r['iqr'])} | {_fmt(s['iqr'])} | {r['n']} | {s['n']} |"
        )
    return "\n".join(lines)


def _md_distance(dist: dict) -> str:
    lines = ["## Distribution distance metrics", ""]
    lines.append("| Quantity | KS statistic | Wasserstein |")
    lines.append("|---|---|---|")
    for q in DISTRIBUTION_QUANTITIES:
        lines.append(
            f"| {q} | {_fmt(dist[q]['ks_statistic'])} | {_fmt(dist[q]['wasserstein'])} |"
        )
    return "\n".join(lines)


def _md_transition(transition_distance: float, seq: dict) -> str:
    lines = ["## AOI transition comparison", ""]
    lines.append(
        f"- Transition-matrix Frobenius distance: **{_fmt(transition_distance)}**"
    )
    lines.append(f"- Bigram Jaccard overlap: {_fmt(seq['bigram_jaccard'])}")
    lines.append(
        f"- Bigram overlap coefficient: {_fmt(seq['bigram_overlap_coefficient'])}"
    )
    lines.append("")
    lines.append("| Rate | Reference | Synthetic |")
    lines.append("|---|---|---|")
    for key in ("regression_rate", "skip_rate", "line_transition_rate"):
        lines.append(
            f"| {key} | {_fmt(seq['reference'][key])} | {_fmt(seq['synthetic'][key])} |"
        )
    return "\n".join(lines)


def _md_scanpaths(plot_paths: dict[str, Path], out_dir: Path) -> str:
    lines = ["## Scanpath examples", ""]
    if not plot_paths:
        lines.append("_No plots generated (insufficient data)._")
        return "\n".join(lines)
    for name in ("fixation_duration_distribution", "saccade_amplitude_distribution",
                 "transition_matrix", "scanpath_examples"):
        p = plot_paths.get(name)
        if p is None:
            continue
        try:
            rel = Path(p).relative_to(out_dir)
        except ValueError:
            rel = Path(p)
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"![{name}]({rel.as_posix()})")
        lines.append("")
    return "\n".join(lines)


def _md_privacy(priv: dict) -> str:
    lines = ["## Privacy proxy metrics", ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Reference records | {priv.get('n_reference_records', 'n/a')} |")
    lines.append(f"| Synthetic records | {priv.get('n_synthetic_records', 'n/a')} |")
    lines.append(f"| Mean NN distance | {_fmt(priv.get('mean_nn_distance'))} |")
    lines.append(f"| Min NN distance | {_fmt(priv.get('min_nn_distance'))} |")
    lines.append(
        f"| Re-identification rate (<= eps) | {_fmt(priv.get('reidentification_rate'))} |"
    )
    if "membership_inference_auc" in priv:
        lines.append(
            f"| Membership-inference AUC | {_fmt(priv['membership_inference_auc'])} |"
        )
    lines.append("")
    lines.append(
        "_Note: in self-comparison mode privacy proxies are degenerate "
        "(synthetic == reference) and should be ignored._"
    )
    return "\n".join(lines)


def _md_verdict(verdict: dict, self_compare: bool) -> str:
    lines = ["## Recommendation on whether synthetic data is usable", ""]
    label = {
        "usable": "USABLE",
        "usable_with_caution": "USABLE WITH CAUTION",
        "not_usable": "NOT USABLE",
        "inconclusive": "INCONCLUSIVE",
    }.get(verdict["verdict"], verdict["verdict"].upper())
    lines.append(f"**Verdict: {label}**")
    lines.append("")
    if self_compare:
        lines.append(
            "_Self-comparison mode: reference and synthetic are identical, so all "
            "distance metrics are ~0 by construction. Use a held-out reference for a "
            "meaningful verdict._"
        )
        lines.append("")
    lines.append(f"- Aggregate KS statistic: {_fmt(verdict['aggregate_ks'])}")
    lines.append(f"- Aggregate Wasserstein: {_fmt(verdict['aggregate_wasserstein'])}")
    lines.append(f"- Transition distance: {_fmt(verdict['transition_distance'])}")
    lines.append("")
    for r in verdict["reasons"]:
        lines.append(f"- {r}")
    return "\n".join(lines)


def _render_markdown(
    *,
    label: str,
    dist: dict,
    seq: dict,
    transition_distance: float,
    priv: dict,
    verdict: dict,
    plot_paths: dict[str, Path],
    out_dir: Path,
    self_compare: bool,
) -> str:
    parts = [
        f"# Quality report: {label}",
        "",
        ("> Self-comparison mode (reference == synthetic)."
         if self_compare else "> Reference vs synthetic comparison."),
        "",
        _md_basic_stats(dist),
        "",
        _md_distance(dist),
        "",
        _md_transition(transition_distance, seq),
        "",
        _md_scanpaths(plot_paths, out_dir),
        "",
        _md_privacy(priv),
        "",
        _md_verdict(verdict, self_compare),
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def build_report(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    label: str = "synthetic",
) -> dict[str, Any]:
    """Compute all metrics, render plots, and write the JSON + Markdown report.

    Supports self-comparison (``reference_df is synthetic_df`` or equal frames),
    the M1 default where no held-out reference exists. Returns the metrics dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"

    self_compare = reference_df is synthetic_df or _frames_equal(
        reference_df, synthetic_df
    )

    dist = distribution_metrics(reference_df, synthetic_df)
    seq = sequence_metrics(reference_df, synthetic_df)
    transition_distance = transition_matrix_distance(reference_df, synthetic_df)
    priv = privacy_metrics(reference_df, synthetic_df)
    verdict = _verdict(dist, transition_distance)

    plot_paths = generate_all_plots(reference_df, synthetic_df, plots_dir)

    metrics: dict[str, Any] = {
        "label": label,
        "self_comparison": self_compare,
        "distribution_metrics": dist,
        "sequence_metrics": seq,
        "transition_matrix_distance": transition_distance,
        "privacy_metrics": priv,
        "verdict": verdict,
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }

    write_json(metrics, out / "quality_report.json")
    md = _render_markdown(
        label=label,
        dist=dist,
        seq=seq,
        transition_distance=transition_distance,
        priv=priv,
        verdict=verdict,
        plot_paths=plot_paths,
        out_dir=out,
        self_compare=self_compare,
    )
    write_text(md, out / "quality_report.md")
    return metrics


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    try:
        return a.equals(b)
    except Exception:
        return False
