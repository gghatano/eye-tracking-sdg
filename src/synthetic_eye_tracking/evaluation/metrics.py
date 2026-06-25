"""Evaluation metrics for synthetic eye-tracking data (spec section 8).

All functions operate purely on events DataFrames matching ``EVENT_COLUMNS``
(see :mod:`synthetic_eye_tracking.schema`). Every function is defensive: empty
inputs, NaNs and single-row groups must never raise.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from ..schema import EventType

# The univariate quantities compared between reference and synthetic samples.
DISTRIBUTION_QUANTITIES: list[str] = [
    "fixation_duration_ms",
    "saccade_duration_ms",
    "saccade_amp_px",
    "saccade_velocity_px_s",
    "scanpath_length",
    "trial_duration",
    "missing_rate",
]

# aoi_id values look like ``L{line}_W{word}`` (e.g. ``L0_W3``).
_AOI_RE = re.compile(r"^L(\d+)_W(\d+)$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clean_numeric(values: Any) -> np.ndarray:
    """Coerce to a 1-D float array with NaN/inf removed."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    return arr


def _iqr(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    q75, q25 = np.percentile(arr, [75, 25])
    return float(q75 - q25)


def _safe_stat(fn, arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    try:
        return float(fn(arr))
    except Exception:
        return float("nan")


def _summary(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": _safe_stat(np.mean, arr),
        "std": _safe_stat(lambda a: np.std(a, ddof=0), arr),
        "median": _safe_stat(np.median, arr),
        "iqr": _iqr(arr),
        "n": int(arr.size),
    }


def _ks(ref: np.ndarray, syn: np.ndarray) -> float:
    if ref.size == 0 or syn.size == 0:
        return float("nan")
    try:
        return float(ks_2samp(ref, syn).statistic)
    except Exception:
        return float("nan")


def _wasserstein(ref: np.ndarray, syn: np.ndarray) -> float:
    if ref.size == 0 or syn.size == 0:
        return float("nan")
    try:
        return float(wasserstein_distance(ref, syn))
    except Exception:
        return float("nan")


def _parse_aoi(aoi_id: Any) -> tuple[int, int] | None:
    """Parse ``L{line}_W{word}`` -> (line, word); None if not parseable."""
    if aoi_id is None or (isinstance(aoi_id, float) and math.isnan(aoi_id)):
        return None
    m = _AOI_RE.match(str(aoi_id))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _fixations(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df is None or len(events_df) == 0:
        return events_df.iloc[0:0] if events_df is not None else pd.DataFrame()
    return events_df[events_df["event_type"] == EventType.FIXATION.value]


# --------------------------------------------------------------------------- #
# Per-quantity extraction
# --------------------------------------------------------------------------- #


def _extract_quantity(events_df: pd.DataFrame, quantity: str) -> np.ndarray:
    """Return the 1-D sample for ``quantity`` from an events DataFrame."""
    if events_df is None or len(events_df) == 0:
        return np.array([], dtype="float64")

    if quantity == "fixation_duration_ms":
        sub = events_df[events_df["event_type"] == EventType.FIXATION.value]
        return _clean_numeric(sub["duration_ms"])
    if quantity == "saccade_duration_ms":
        sub = events_df[events_df["event_type"] == EventType.SACCADE.value]
        return _clean_numeric(sub["duration_ms"])
    if quantity == "saccade_amp_px":
        return _clean_numeric(events_df["saccade_amp_px"])
    if quantity == "saccade_velocity_px_s":
        return _clean_numeric(events_df["saccade_velocity_px_s"])
    if quantity == "scanpath_length":
        sub = events_df.copy()
        sub["saccade_amp_px"] = pd.to_numeric(sub["saccade_amp_px"], errors="coerce")
        grouped = sub.groupby(["subject_id", "trial_id"], dropna=False)["saccade_amp_px"]
        return _clean_numeric(grouped.sum(min_count=1))
    if quantity == "trial_duration":
        sub = events_df.copy()
        sub["start_time_ms"] = pd.to_numeric(sub["start_time_ms"], errors="coerce")
        sub["end_time_ms"] = pd.to_numeric(sub["end_time_ms"], errors="coerce")
        grp = sub.groupby(["subject_id", "trial_id"], dropna=False)
        dur = grp["end_time_ms"].max() - grp["start_time_ms"].min()
        return _clean_numeric(dur)
    if quantity == "missing_rate":
        # Fraction of validity==0 events per subject x trial.
        sub = events_df.copy()
        sub["_missing"] = (pd.to_numeric(sub["validity"], errors="coerce") == 0).astype(
            "float64"
        )
        grouped = sub.groupby(["subject_id", "trial_id"], dropna=False)["_missing"]
        return _clean_numeric(grouped.mean())

    raise ValueError(f"unknown quantity: {quantity!r}")


# --------------------------------------------------------------------------- #
# 1. Distribution metrics
# --------------------------------------------------------------------------- #


def distribution_metrics(
    reference_df: pd.DataFrame, synthetic_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Compare reference vs synthetic distributions for each tracked quantity.

    Returns a nested dict keyed by quantity, each value containing ``reference``
    and ``synthetic`` summary stats plus the cross-sample ``ks_statistic`` and
    ``wasserstein`` distance.
    """
    out: dict[str, dict[str, Any]] = {}
    for q in DISTRIBUTION_QUANTITIES:
        ref = _extract_quantity(reference_df, q)
        syn = _extract_quantity(synthetic_df, q)
        out[q] = {
            "reference": _summary(ref),
            "synthetic": _summary(syn),
            "ks_statistic": _ks(ref, syn),
            "wasserstein": _wasserstein(ref, syn),
        }
    return out


# --------------------------------------------------------------------------- #
# 2/3. Transition matrices
# --------------------------------------------------------------------------- #


def transition_matrix(events_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a normalized AOI->AOI transition matrix over consecutive fixations.

    Transitions are counted within each ``subject_id`` x ``trial_id`` group, in
    ``event_index`` order. Null/unparseable AOIs are skipped. Returns the
    row-normalized probability matrix (rows sum to 1 where the source AOI has any
    outgoing transitions) and the ordered list of AOI labels.
    """
    fix = _fixations(events_df)
    counts: dict[tuple[str, str], int] = {}
    labels: set[str] = set()

    if fix is not None and len(fix) > 0:
        fix = fix.copy()
        fix["event_index"] = pd.to_numeric(fix["event_index"], errors="coerce")
        for _, grp in fix.groupby(["subject_id", "trial_id"], dropna=False):
            grp = grp.sort_values("event_index")
            seq = [a for a in grp["aoi_id"].tolist() if _valid_aoi(a)]
            for a, b in zip(seq[:-1], seq[1:]):
                a, b = str(a), str(b)
                labels.add(a)
                labels.add(b)
                counts[(a, b)] = counts.get((a, b), 0) + 1

    ordered = sorted(labels)
    mat = pd.DataFrame(0.0, index=ordered, columns=ordered)
    for (a, b), c in counts.items():
        mat.at[a, b] = float(c)

    row_sums = mat.sum(axis=1)
    for a in ordered:
        if row_sums[a] > 0:
            mat.loc[a] = mat.loc[a] / row_sums[a]
    return mat, ordered


def _valid_aoi(a: Any) -> bool:
    if a is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return False
    s = str(a).strip()
    return s != "" and s.lower() != "nan"


def transition_matrix_distance(ref_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    """Frobenius norm between probability matrices aligned on the AOI union."""
    ref_mat, ref_labels = transition_matrix(ref_df)
    syn_mat, syn_labels = transition_matrix(syn_df)
    union = sorted(set(ref_labels) | set(syn_labels))
    if not union:
        return float("nan")
    ref_a = ref_mat.reindex(index=union, columns=union, fill_value=0.0)
    syn_a = syn_mat.reindex(index=union, columns=union, fill_value=0.0)
    diff = ref_a.to_numpy(dtype="float64") - syn_a.to_numpy(dtype="float64")
    return float(np.linalg.norm(diff, ord="fro"))


# --------------------------------------------------------------------------- #
# 4. Sequence metrics
# --------------------------------------------------------------------------- #


def _aoi_sequences(events_df: pd.DataFrame) -> list[list[str]]:
    """Per subject x trial ordered list of valid AOI labels (fixations only)."""
    fix = _fixations(events_df)
    sequences: list[list[str]] = []
    if fix is None or len(fix) == 0:
        return sequences
    fix = fix.copy()
    fix["event_index"] = pd.to_numeric(fix["event_index"], errors="coerce")
    for _, grp in fix.groupby(["subject_id", "trial_id"], dropna=False):
        grp = grp.sort_values("event_index")
        seq = [str(a) for a in grp["aoi_id"].tolist() if _valid_aoi(a)]
        if seq:
            sequences.append(seq)
    return sequences


def _bigrams(sequences: list[list[str]]) -> set[tuple[str, str]]:
    grams: set[tuple[str, str]] = set()
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            grams.add((a, b))
    return grams


def _sequence_rates(events_df: pd.DataFrame) -> dict[str, float]:
    """Regression / skip / line-transition rates from parsed AOI indices."""
    sequences = _aoi_sequences(events_df)
    total = 0
    regressions = 0
    skips = 0
    line_transitions = 0
    for seq in sequences:
        parsed = [_parse_aoi(a) for a in seq]
        for prev, cur in zip(parsed[:-1], parsed[1:]):
            if prev is None or cur is None:
                continue
            total += 1
            p_line, p_word = prev
            c_line, c_word = cur
            if c_line != p_line:
                line_transitions += 1
                continue
            # Same line: judge by word index movement.
            delta = c_word - p_word
            if delta < 0:
                regressions += 1
            elif delta > 1:
                skips += 1
    if total == 0:
        return {"regression_rate": float("nan"), "skip_rate": float("nan"),
                "line_transition_rate": float("nan")}
    return {
        "regression_rate": regressions / total,
        "skip_rate": skips / total,
        "line_transition_rate": line_transitions / total,
    }


def sequence_metrics(ref_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    """Sequence-level comparison: bigram overlap + reading-pattern rates."""
    ref_seq = _aoi_sequences(ref_df)
    syn_seq = _aoi_sequences(syn_df)
    ref_grams = _bigrams(ref_seq)
    syn_grams = _bigrams(syn_seq)

    inter = len(ref_grams & syn_grams)
    union = len(ref_grams | syn_grams)
    jaccard = (inter / union) if union else float("nan")
    smaller = min(len(ref_grams), len(syn_grams))
    overlap_coeff = (inter / smaller) if smaller else float("nan")

    ref_rates = _sequence_rates(ref_df)
    syn_rates = _sequence_rates(syn_df)

    return {
        "bigram_jaccard": jaccard,
        "bigram_overlap_coefficient": overlap_coeff,
        "n_reference_bigrams": len(ref_grams),
        "n_synthetic_bigrams": len(syn_grams),
        "reference": ref_rates,
        "synthetic": syn_rates,
    }


# --------------------------------------------------------------------------- #
# 5. Privacy metrics (spec 8.4)
# --------------------------------------------------------------------------- #


def _feature_table(events_df: pd.DataFrame) -> np.ndarray:
    """Per subject x trial feature vector: [mean fix dur, mean sacc amp, n events,
    missing rate]. Returns an (n_records, 4) float array."""
    if events_df is None or len(events_df) == 0:
        return np.empty((0, 4), dtype="float64")
    df = events_df.copy()
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    df["saccade_amp_px"] = pd.to_numeric(df["saccade_amp_px"], errors="coerce")
    df["validity"] = pd.to_numeric(df["validity"], errors="coerce")
    df["_is_fix"] = df["event_type"] == EventType.FIXATION.value
    rows = []
    for _, grp in df.groupby(["subject_id", "trial_id"], dropna=False):
        fix_dur = grp.loc[grp["_is_fix"], "duration_ms"].mean()
        sacc_amp = grp["saccade_amp_px"].mean()
        n_events = float(len(grp))
        missing = float((grp["validity"] == 0).mean())
        rows.append([fix_dur, sacc_amp, n_events, missing])
    arr = np.array(rows, dtype="float64")
    # Replace NaNs with column means (then 0 if a whole column is NaN).
    if arr.size:
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(~np.isfinite(arr))
        arr[inds] = np.take(col_means, inds[1])
    return arr


def _standardize(ref: np.ndarray, syn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both with the reference mean/std (avoids scale dominance)."""
    if ref.size == 0:
        return ref, syn
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (ref - mu) / sd, (syn - mu) / sd


def privacy_metrics(
    ref_df: pd.DataFrame, syn_df: pd.DataFrame, *, epsilon: float = 0.1
) -> dict[str, Any]:
    """Simple privacy proxies (spec 8.4).

    - ``mean_nn_distance``: mean nearest-neighbour distance from each synthetic
      record to the closest reference record (standardized feature space).
    - ``reidentification_rate``: fraction of synthetic records with NN distance
      below ``epsilon`` (potential near-copies).
    - ``membership_inference_auc``: crude MI proxy (1-NN distance classifier
      distinguishing reference-train vs holdout); omitted gracefully if infeasible.
    """
    ref_feat = _feature_table(ref_df)
    syn_feat = _feature_table(syn_df)
    out: dict[str, Any] = {
        "n_reference_records": int(ref_feat.shape[0]),
        "n_synthetic_records": int(syn_feat.shape[0]),
    }
    if ref_feat.shape[0] == 0 or syn_feat.shape[0] == 0:
        out["mean_nn_distance"] = float("nan")
        out["min_nn_distance"] = float("nan")
        out["reidentification_rate"] = float("nan")
        return out

    ref_s, syn_s = _standardize(ref_feat, syn_feat)
    # Pairwise distances syn -> ref.
    dists = np.linalg.norm(
        syn_s[:, None, :] - ref_s[None, :, :], axis=2
    )  # (n_syn, n_ref)
    nn = dists.min(axis=1)
    out["mean_nn_distance"] = float(np.mean(nn))
    out["min_nn_distance"] = float(np.min(nn))
    out["reidentification_rate"] = float(np.mean(nn < epsilon))

    mi = _membership_inference_auc(ref_feat, syn_feat)
    if mi is not None:
        out["membership_inference_auc"] = mi
    return out


def _membership_inference_auc(
    ref_feat: np.ndarray, syn_feat: np.ndarray
) -> float | None:
    """Crude MI proxy: split reference into member/non-member, score each by NN
    distance to the synthetic set, and report AUC. ~0.5 means low leakage.

    Returns None when there are too few records to be meaningful.
    """
    n = ref_feat.shape[0]
    if n < 4 or syn_feat.shape[0] < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    half = n // 2
    members = ref_feat[:half]
    nonmembers = ref_feat[half : 2 * half]
    if members.shape[0] == 0 or nonmembers.shape[0] == 0:
        return None

    ref_s, syn_s = _standardize(ref_feat, syn_feat)
    mem_s = ref_s[:half]
    non_s = ref_s[half : 2 * half]

    def _nn(a: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(a[:, None, :] - syn_s[None, :, :], axis=2)
        return d.min(axis=1)

    mem_d = _nn(mem_s)
    non_d = _nn(non_s)
    # Members should be *closer* to synthetic if there is leakage -> use -dist
    # as the "membership" score.
    scores = np.concatenate([-mem_d, -non_d])
    labels = np.concatenate([np.ones(mem_d.size), np.zeros(non_d.size)])
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None
