"""Generate figures and summary stats for the data/model report (docs/).

Runs both generators on a moderate config, writes PNGs to docs/assets/ and a
stats JSON used to fill in numbers in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_eye_tracking.config import Config, ExperimentConfig
from synthetic_eye_tracking.generators.model_based import ModelBasedGenerator
from synthetic_eye_tracking.generators.rule_based import RuleBasedGenerator

ASSETS = Path("docs/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)

RULE = "#2563eb"   # blue
MODEL = "#dc2626"  # red
ACCENT = "#059669" # green


def save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    cfg = Config(
        experiment=ExperimentConfig(
            n_subjects=40, n_stimuli=5, trials_per_subject=6, seed=42
        )
    )

    # ---- Rule-based ----
    rb = RuleBasedGenerator(cfg)
    rb_events = rb.generate_events()
    rb_raw = rb.generate_raw_gaze(rb_events)
    participants = rb.participants
    aoi_layout = rb.aoi_layout

    # ---- Model-based (fit on rule-based as pseudo-reference) ----
    mb = ModelBasedGenerator(cfg)
    mb.fit(rb_events)
    mb_events = mb.generate_events(
        n_subjects=cfg.experiment.n_subjects, n_trials=cfg.experiment.trials_per_subject
    )

    stats: dict = {}

    # ---------------------------------------------------------------- #
    # 1. AOI reading-grid layout (one stimulus)
    # ---------------------------------------------------------------- #
    one = aoi_layout[aoi_layout["stimulus_id"] == aoi_layout["stimulus_id"].iloc[0]]
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in one.itertuples(index=False):
        ax.add_patch(
            plt.Rectangle(
                (r.x_min, r.y_min),
                r.x_max - r.x_min,
                r.y_max - r.y_min,
                facecolor=plt.cm.viridis(r.difficulty),
                edgecolor="white",
                linewidth=0.6,
            )
        )
    ax.set_xlim(0, cfg.screen.width_px)
    ax.set_ylim(cfg.screen.height_px, 0)  # screen coords: y down
    ax.set_title("Reading-grid AOI layout (color = per-AOI difficulty)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([0, 1])
    fig.colorbar(sm, ax=ax, label="difficulty", fraction=0.025)
    save(fig, "aoi_layout.png")

    # ---------------------------------------------------------------- #
    # 2. Example scanpaths: fast/low-regression vs slow/high-regression reader
    # ---------------------------------------------------------------- #
    p = participants.copy()
    fast = p.sort_values("reading_speed_factor", ascending=False).iloc[0]["subject_id"]
    regr = p.sort_values("regression_tendency", ascending=False).iloc[0]["subject_id"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, sid, title in [
        (axes[0], fast, f"Fast reader ({fast})"),
        (axes[1], regr, f"High-regression reader ({regr})"),
    ]:
        sub = rb_events[(rb_events["subject_id"] == sid)]
        # pick that subject's first trial
        tid = sub["trial_id"].iloc[0]
        fx = sub[(sub["trial_id"] == tid) & (sub["event_type"] == "fixation")]
        stim = fx["stimulus_id"].iloc[0]
        lay = aoi_layout[aoi_layout["stimulus_id"] == stim]
        ax.scatter(lay["center_x"], lay["center_y"], s=8, c="#cbd5e1", zorder=1)
        ax.plot(fx["x_px"], fx["y_px"], "-o", color=RULE, ms=4, lw=0.9, zorder=2)
        ax.scatter(
            fx["x_px"].iloc[0], fx["y_px"].iloc[0], c=ACCENT, s=60, zorder=3, label="start"
        )
        ax.set_title(title)
        ax.set_xlabel("x (px)")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("y (px)")
    fig.suptitle("Example scanpaths — individual diversity in reading behavior")
    save(fig, "scanpaths.png")

    # ---------------------------------------------------------------- #
    # 3. Fixation duration distribution: rule vs model
    # ---------------------------------------------------------------- #
    rb_fix = rb_events.loc[rb_events.event_type == "fixation", "duration_ms"].dropna()
    mb_fix = mb_events.loc[mb_events.event_type == "fixation", "duration_ms"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 800, 50)
    ax.hist(rb_fix, bins=bins, density=True, alpha=0.55, color=RULE, label="rule-based")
    ax.hist(mb_fix, bins=bins, density=True, alpha=0.55, color=MODEL, label="model-based")
    ax.axvline(240, color="k", ls="--", lw=1, label="config mean (240 ms)")
    ax.set_title("Fixation duration distribution (right-skewed, lognormal-like)")
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("density")
    ax.legend()
    save(fig, "fixation_duration.png")

    # ---------------------------------------------------------------- #
    # 4. Saccade amplitude distribution
    # ---------------------------------------------------------------- #
    rb_amp = rb_events["saccade_amp_px"].dropna()
    mb_amp = mb_events["saccade_amp_px"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(rb_amp.max(), mb_amp.max()), 50)
    ax.hist(rb_amp, bins=bins, density=True, alpha=0.55, color=RULE, label="rule-based")
    ax.hist(mb_amp, bins=bins, density=True, alpha=0.55, color=MODEL, label="model-based")
    ax.set_title("Saccade amplitude distribution")
    ax.set_xlabel("amplitude (px)")
    ax.set_ylabel("density")
    ax.legend()
    save(fig, "saccade_amplitude.png")

    # ---------------------------------------------------------------- #
    # 5. Raw gaze time-series sample (x over time) with blink/missing gaps
    # ---------------------------------------------------------------- #
    one_raw = rb_raw[
        (rb_raw.subject_id == rb_raw.subject_id.iloc[0])
        & (rb_raw.trial_id == rb_raw.trial_id.iloc[0])
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(one_raw["timestamp_ms"], one_raw["x_px"], color=RULE, lw=0.8, label="x")
    ax.plot(one_raw["timestamp_ms"], one_raw["y_px"], color=ACCENT, lw=0.8, label="y")
    miss = one_raw[one_raw["validity"] == 0]
    for t in miss["timestamp_ms"]:
        ax.axvline(t, color="#fca5a5", lw=0.4, alpha=0.5)
    ax.set_title(f"Raw gaze time-series @ {cfg.sampling.frequency_hz:.0f} Hz "
                 f"(red = blink/missing samples)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("position (px)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "raw_gaze.png")

    # ---------------------------------------------------------------- #
    # 6. AOI transition matrix (rule-based) heatmap
    # ---------------------------------------------------------------- #
    from synthetic_eye_tracking.evaluation.metrics import transition_matrix

    tm, labels = transition_matrix(rb_events)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(tm.values, cmap="magma", aspect="auto")
    ax.set_title("AOI→AOI transition probability (rule-based)")
    ax.set_xlabel("to AOI (index)")
    ax.set_ylabel("from AOI (index)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(to | from)")
    save(fig, "transition_matrix.png")

    # ---------------------------------------------------------------- #
    # 7. Individual diversity: participant trait distributions
    # ---------------------------------------------------------------- #
    traits = [
        ("reading_speed_factor", "Reading speed factor"),
        ("regression_tendency", "Regression tendency"),
        ("fixation_noise_sd", "Fixation noise SD (px)"),
        ("missing_rate", "Missing/blink rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (col, title) in zip(axes.ravel(), traits):
        ax.hist(participants[col], bins=15, color=ACCENT, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("# participants")
    fig.suptitle("Participant-level parameterization → individual diversity")
    save(fig, "participant_traits.png")

    # ---------------------------------------------------------------- #
    # 8. Diversity DOES drive outcomes: trait vs realized behavior
    # ---------------------------------------------------------------- #
    # per-subject realized: mean fixation duration & regression rate
    fix = rb_events[rb_events.event_type == "fixation"]
    mean_fix = fix.groupby("subject_id")["duration_ms"].mean()

    def regression_rate(df: pd.DataFrame) -> float:
        sac = df[df.event_type == "saccade"]
        if len(sac) == 0:
            return np.nan
        # leftward / upward saccades approximate regressions
        return float((sac["saccade_angle_deg"].abs() > 120).mean())

    rr = rb_events.groupby("subject_id").apply(regression_rate, include_groups=False)
    merged = participants.set_index("subject_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].scatter(merged["reading_speed_factor"], mean_fix.reindex(merged.index),
                    c=RULE, alpha=0.7)
    axes[0].set_xlabel("reading_speed_factor (parameter)")
    axes[0].set_ylabel("mean fixation duration (ms, realized)")
    axes[0].set_title("Faster reading speed → shorter fixations")
    axes[1].scatter(merged["regression_tendency"], rr.reindex(merged.index),
                    c=MODEL, alpha=0.7)
    axes[1].set_xlabel("regression_tendency (parameter)")
    axes[1].set_ylabel("realized regression rate")
    axes[1].set_title("Higher regression tendency → more regressions")
    fig.suptitle("Parameters map to observable behavioral diversity")
    save(fig, "diversity_effect.png")

    # ---------------------------------------------------------------- #
    # Summary stats for the report
    # ---------------------------------------------------------------- #
    from synthetic_eye_tracking.evaluation.metrics import (
        distribution_metrics,
        transition_matrix_distance,
    )

    dm = distribution_metrics(rb_events, mb_events)
    stats["config"] = {
        "n_subjects": cfg.experiment.n_subjects,
        "n_stimuli": cfg.experiment.n_stimuli,
        "trials_per_subject": cfg.experiment.trials_per_subject,
        "frequency_hz": cfg.sampling.frequency_hz,
    }
    stats["counts"] = {
        "rule_events": int(len(rb_events)),
        "model_events": int(len(mb_events)),
        "rule_raw_samples": int(len(rb_raw)),
        "n_participants": int(len(participants)),
        "n_aoi_per_stimulus": int(
            aoi_layout.groupby("stimulus_id").size().iloc[0]
        ),
    }
    stats["fixation_duration"] = {
        "rule_mean": round(float(rb_fix.mean()), 1),
        "rule_median": round(float(rb_fix.median()), 1),
        "model_mean": round(float(mb_fix.mean()), 1),
        "ks": round(float(dm["fixation_duration_ms"]["ks_statistic"]), 3),
        "wasserstein": round(float(dm["fixation_duration_ms"]["wasserstein"]), 2),
    }
    stats["saccade_amp"] = {
        "rule_mean": round(float(rb_amp.mean()), 1),
        "ks": round(float(dm["saccade_amp_px"]["ks_statistic"]), 3),
    }
    stats["transition_matrix_distance"] = round(
        float(transition_matrix_distance(rb_events, mb_events)), 3
    )
    stats["participant_trait_ranges"] = {
        col: [round(float(participants[col].min()), 3),
              round(float(participants[col].max()), 3)]
        for col, _ in traits
    }

    Path("docs/_report_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nSTATS:\n", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
