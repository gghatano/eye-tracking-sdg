"""Sim-to-real event-detection experiment (M3-A).

Trains a fixation-vs-saccade classifier and compares three regimes on an
identical real test set (subject/series-grouped folds of Lund2013):

* TSTR  - train on synthetic only, test on real (the contribution test)
* TRTR  - train on real, test on real (the practical upper bound)
* R+S   - train on real + synthetic, test on real (augmentation)

The honest question is whether TSTR approaches TRTR. If it does, synthetic data
can stand in for scarce labelled real data on this task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeResult:
    name: str
    f1_macro: float
    balanced_accuracy: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]
    labels: list[str]
    n_train: int
    n_test: int


@dataclass
class ExperimentResult:
    regimes: dict[str, RegimeResult]
    real_label_counts: dict[str, int]
    synth_label_counts: dict[str, int]
    feature_names: list[str]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "regimes": {
                k: {
                    "name": v.name,
                    "f1_macro": v.f1_macro,
                    "balanced_accuracy": v.balanced_accuracy,
                    "per_class_f1": v.per_class_f1,
                    "confusion": v.confusion,
                    "labels": v.labels,
                    "n_train": v.n_train,
                    "n_test": v.n_test,
                }
                for k, v in self.regimes.items()
            },
            "real_label_counts": self.real_label_counts,
            "synth_label_counts": self.synth_label_counts,
            "feature_names": self.feature_names,
            "meta": self.meta,
        }


def _make_classifier(seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _score(y_true, y_pred, labels: list[str]) -> tuple[float, float, dict, list]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )

    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {lab: float(v) for lab, v in zip(labels, per_class)}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return f1_macro, bal, per_class_f1, cm


def run_experiment(
    *,
    n_domains: int = 8,
    real_annotator: str = "RA",
    real_categories: tuple[str, ...] = ("img", "dots", "video"),
    n_splits: int = 4,
    window: int = 7,
    seed: int = 0,
    synth_subjects: int = 6,
    synth_trials: int = 4,
    synth_stimuli: int = 2,
    frequency_hz: float = 500.0,
) -> ExperimentResult:
    """Run the sim-to-real comparison and return aggregated metrics."""
    from sklearn.model_selection import GroupKFold

    from .features import extract_features, filter_scored
    from .lund2013 import load_lund2013_samples
    from .synthetic import generate_synthetic_samples

    # ---- real ----
    real = load_lund2013_samples(annotator=real_annotator, categories=real_categories)
    real_counts = {k: int(v) for k, v in real["label"].value_counts().items()}
    real_fs = filter_scored(extract_features(real, window=window))

    # ---- synthetic (domain-randomized) ----
    synth = generate_synthetic_samples(
        n_domains=n_domains,
        seed=seed,
        frequency_hz=frequency_hz,
        n_subjects=synth_subjects,
        trials_per_subject=synth_trials,
        n_stimuli=synth_stimuli,
    )
    synth_counts = {k: int(v) for k, v in synth["label"].value_counts().items()}
    synth_fs = filter_scored(extract_features(synth, window=window))

    labels = sorted(set(real_fs.y) | set(synth_fs.y))

    Xr, yr, gr = real_fs.X, real_fs.y, real_fs.groups
    Xs, ys = synth_fs.X, synth_fs.y

    # Accumulate predictions across folds so every regime is scored on the same
    # held-out real samples.
    acc: dict[str, dict[str, list]] = {
        r: {"true": [], "pred": [], "n_train": 0} for r in ("TSTR", "TRTR", "R+S")
    }

    n_groups = len(np.unique(gr))
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    for tr_idx, te_idx in gkf.split(Xr, yr, groups=gr):
        Xr_tr, yr_tr = Xr[tr_idx], yr[tr_idx]
        Xr_te, yr_te = Xr[te_idx], yr[te_idx]

        # TSTR: synthetic -> real test
        clf = _make_classifier(seed)
        clf.fit(Xs, ys)
        acc["TSTR"]["true"].append(yr_te)
        acc["TSTR"]["pred"].append(clf.predict(Xr_te))
        acc["TSTR"]["n_train"] = len(ys)

        # TRTR: real train -> real test
        clf = _make_classifier(seed)
        clf.fit(Xr_tr, yr_tr)
        acc["TRTR"]["true"].append(yr_te)
        acc["TRTR"]["pred"].append(clf.predict(Xr_te))
        acc["TRTR"]["n_train"] += len(yr_tr)

        # R+S: real train + synthetic -> real test
        clf = _make_classifier(seed)
        X_rs = np.vstack([Xr_tr, Xs])
        y_rs = np.concatenate([yr_tr, ys])
        clf.fit(X_rs, y_rs)
        acc["R+S"]["true"].append(yr_te)
        acc["R+S"]["pred"].append(clf.predict(Xr_te))
        acc["R+S"]["n_train"] += len(y_rs)

    regimes: dict[str, RegimeResult] = {}
    for name, d in acc.items():
        y_true = np.concatenate(d["true"])
        y_pred = np.concatenate(d["pred"])
        f1m, bal, pcf1, cm = _score(y_true, y_pred, labels)
        regimes[name] = RegimeResult(
            name=name,
            f1_macro=f1m,
            balanced_accuracy=bal,
            per_class_f1=pcf1,
            confusion=cm,
            labels=labels,
            n_train=int(d["n_train"]),
            n_test=int(len(y_true)),
        )

    return ExperimentResult(
        regimes=regimes,
        real_label_counts=real_counts,
        synth_label_counts=synth_counts,
        feature_names=list(real_fs.feature_names),
        meta={
            "n_domains": n_domains,
            "n_splits": splits,
            "window": window,
            "seed": seed,
            "frequency_hz": frequency_hz,
            "real_annotator": real_annotator,
            "real_categories": list(real_categories),
        },
    )
