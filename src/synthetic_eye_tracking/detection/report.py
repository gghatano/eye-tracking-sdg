"""Report and figures for the sim-to-real event-detection experiment."""

from __future__ import annotations

from pathlib import Path

from .experiment import ExperimentResult


def _plot_confusions(result: ExperimentResult, out_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: dict[str, Path] = {}
    for name, reg in result.regimes.items():
        cm = np.array(reg.confusion, dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(reg.labels)), reg.labels, rotation=45, ha="right")
        ax.set_yticks(range(len(reg.labels)), reg.labels)
        for i in range(len(reg.labels)):
            for j in range(len(reg.labels)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name} (row-normalized)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        safe = name.replace("+", "plus").replace("/", "_")
        p = out_dir / f"confusion_{safe}.png"
        fig.savefig(p, bbox_inches="tight", dpi=110)
        plt.close(fig)
        paths[name] = p
    return paths


def _plot_f1_bars(result: ExperimentResult, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(result.regimes.keys())
    f1s = [result.regimes[n].f1_macro for n in names]
    bals = [result.regimes[n].balanced_accuracy for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x - w / 2, f1s, w, label="macro F1", color="#2563eb")
    ax.bar(x + w / 2, bals, w, label="balanced acc.", color="#059669")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Event detection: train regime comparison (real test set)")
    for i, v in enumerate(f1s):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    p = out_dir / "regime_comparison.png"
    fig.savefig(p, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return p


def build_detection_report(result: ExperimentResult, out_dir: str | Path) -> dict:
    """Write quality_report.json/md and figures for the experiment."""
    import json

    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    cm_paths = _plot_confusions(result, out / "plots")
    bar_path = _plot_f1_bars(result, out / "plots")

    (out / "detection_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str)
    )

    reg = result.regimes
    tstr = reg.get("TSTR")
    trtr = reg.get("TRTR")
    gap = (trtr.f1_macro - tstr.f1_macro) if (tstr and trtr) else float("nan")
    retention = (tstr.f1_macro / trtr.f1_macro) if (tstr and trtr and trtr.f1_macro) else float("nan")

    lines: list[str] = []
    lines.append("# Sim-to-real event detection: results\n")
    lines.append(
        "Fixation-vs-saccade detection. Real data: Lund2013 (Andersson et al. 2017), "
        "expert hand-labelled, free-viewing. Synthetic training data comes from this "
        "project's rule-based generator under domain randomization. All regimes are "
        "scored on the same series-grouped held-out real folds.\n"
    )

    lines.append("## Headline\n")
    if tstr and trtr:
        lines.append(
            f"- Train-on-synthetic, test-on-real (TSTR) macro F1: **{tstr.f1_macro:.3f}**\n"
            f"- Train-on-real, test-on-real (TRTR) macro F1: **{trtr.f1_macro:.3f}**\n"
            f"- Gap (TRTR - TSTR): **{gap:.3f}**; TSTR retains **{retention*100:.0f}%** of the real-trained F1\n"
        )

    lines.append("\n## Per-regime metrics\n")
    header = "| regime | macro F1 | balanced acc. | " + " | ".join(
        f"F1[{l}]" for l in (tstr.labels if tstr else [])
    ) + " | n_train | n_test |"
    sep = "|" + "---|" * (4 + (len(tstr.labels) if tstr else 0) + 1)
    lines.append(header)
    lines.append(sep)
    for name, r in reg.items():
        pcf = " | ".join(f"{r.per_class_f1.get(l, float('nan')):.3f}" for l in r.labels)
        lines.append(
            f"| {name} | {r.f1_macro:.3f} | {r.balanced_accuracy:.3f} | {pcf} | {r.n_train} | {r.n_test} |"
        )

    lines.append("\n## Regime comparison\n")
    lines.append(f"![regime comparison](plots/{bar_path.name})\n")
    lines.append("\n## Confusion matrices\n")
    for name, p in cm_paths.items():
        lines.append(f"### {name}\n")
        lines.append(f"![confusion {name}](plots/{p.name})\n")

    lines.append("\n## Label distribution\n")
    lines.append("| label | real samples | synthetic samples |")
    lines.append("|---|---|---|")
    all_labels = sorted(set(result.real_label_counts) | set(result.synth_label_counts))
    for l in all_labels:
        lines.append(
            f"| {l} | {result.real_label_counts.get(l, 0)} | {result.synth_label_counts.get(l, 0)} |"
        )

    lines.append("\n## Interpretation and caveats\n")
    lines.append(
        "- A small TRTR-minus-TSTR gap means synthetic data, with its free perfect "
        "labels, can substitute for scarce hand-labelled real data on this task. "
        "Adding synthetic data to real (R+S) tests whether augmentation helps.\n"
        "- **Domain.** Lund2013 is free-viewing of images, dots, and video, not "
        "reading. This demonstrates general-domain transfer; a reading-specific claim "
        "needs a reading dataset with raw samples and event labels, which was not "
        "reachable in this environment.\n"
        "- **Excluded classes.** Smooth pursuit and post-saccadic oscillation exist in "
        "the real data but not in the synthetic generator, so they are excluded from "
        "this fixation-vs-saccade comparison rather than penalised.\n"
        "- **Features.** Both sources pass through the same degree-of-visual-angle, "
        "velocity-based feature extractor, which is what makes cross-device transfer "
        "meaningful.\n"
    )

    md = "\n".join(lines) + "\n"
    (out / "detection_report.md").write_text(md)
    return result.to_dict()
