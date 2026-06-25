"""Command-line interface (spec section 9).

Subcommands:
  generate-rule    Rule-based generation (M1)
  fit-model        Fit a model-based generator to reference data (M2)
  generate-model   Generate from a fitted model (M2)
  evaluate         Compare reference vs synthetic and write a report
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config

app = typer.Typer(
    add_completion=False,
    help="Synthetic eye-tracking data generator (rule-based + model-based).",
    no_args_is_help=True,
)


@app.command("generate-rule")
def generate_rule(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    output: Path = typer.Option(..., "--output", help="Output directory."),
) -> None:
    """Generate synthetic data with the rule-based generator."""
    from .evaluation.report import build_report
    from .generators.rule_based import RuleBasedGenerator
    from .io.writers import write_generation_bundle

    cfg = load_config(config)
    typer.echo(f"[generate-rule] config={config} -> output={output}")

    gen = RuleBasedGenerator(cfg)
    events = gen.generate_events()
    raw_gaze = gen.generate_raw_gaze(events)
    aoi_summary = gen.generate_aoi_summary(events)

    written = write_generation_bundle(
        output,
        events=events,
        raw_gaze=raw_gaze,
        aoi_summary=aoi_summary,
        participants=getattr(gen, "participants", None),
        stimuli=getattr(gen, "stimuli", None),
        aoi_layout=getattr(gen, "aoi_layout", None),
        config=cfg,
    )
    typer.echo(f"[generate-rule] wrote {len(written)} files, {len(events)} events.")

    # M1: no external reference -> self-consistency report.
    build_report(events, events, output, label="rule_based")
    typer.echo(f"[generate-rule] quality report -> {output}/quality_report.md")


@app.command("fit-model")
def fit_model(
    config: Path = typer.Option(..., "--config"),
    input: Path = typer.Option(..., "--input", help="Reference events CSV."),
    model_output: Path = typer.Option(..., "--model-output", help="Output .pkl path."),
) -> None:
    """Fit a model-based generator to reference data (M2)."""
    from .generators.model_based import ModelBasedGenerator
    from .io.loaders import load_events

    cfg = load_config(config)
    typer.echo(f"[fit-model] input={input} -> model={model_output}")
    reference = load_events(input)
    gen = ModelBasedGenerator(cfg)
    gen.fit(reference)
    gen.save(model_output)
    typer.echo(f"[fit-model] saved model -> {model_output}")


@app.command("generate-model")
def generate_model(
    config: Path = typer.Option(..., "--config"),
    model: Path = typer.Option(..., "--model", help="Fitted model .pkl path."),
    output: Path = typer.Option(..., "--output", help="Output directory."),
) -> None:
    """Generate synthetic data from a fitted model (M2)."""
    from .evaluation.report import build_report
    from .generators.model_based import ModelBasedGenerator
    from .io.writers import write_generation_bundle

    cfg = load_config(config)
    typer.echo(f"[generate-model] model={model} -> output={output}")

    gen = ModelBasedGenerator.load(model, config=cfg)
    events = gen.generate_events(
        n_subjects=cfg.experiment.n_subjects,
        n_trials=cfg.experiment.trials_per_subject,
    )
    raw_gaze = gen.generate_raw_gaze(events)
    aoi_summary = gen.generate_aoi_summary(events)

    write_generation_bundle(
        output,
        events=events,
        raw_gaze=raw_gaze,
        aoi_summary=aoi_summary,
        config=cfg,
    )
    build_report(events, events, output, label="model_based")
    typer.echo(f"[generate-model] wrote {len(events)} events -> {output}")


@app.command("evaluate")
def evaluate(
    reference: Path = typer.Option(..., "--reference", help="Reference events CSV."),
    synthetic: Path = typer.Option(..., "--synthetic", help="Synthetic events CSV."),
    output: Path = typer.Option(..., "--output", help="Report output directory."),
) -> None:
    """Compare reference vs synthetic events and write a quality report."""
    from .evaluation.report import build_report
    from .io.loaders import load_events

    typer.echo(f"[evaluate] reference={reference} synthetic={synthetic} -> {output}")
    ref = load_events(reference)
    syn = load_events(synthetic)
    build_report(ref, syn, output, label="comparison")
    typer.echo(f"[evaluate] report -> {output}/quality_report.md")


@app.command("detect-events")
def detect_events(
    output: Path = typer.Option(..., "--output", help="Report output directory."),
    n_domains: int = typer.Option(8, "--n-domains", help="Domain-randomized synthetic setups."),
    n_splits: int = typer.Option(4, "--n-splits", help="Series-grouped CV folds on real data."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Sim-to-real fixation/saccade detection (M3-A): train on synthetic, test on real."""
    from .detection.experiment import run_experiment
    from .detection.report import build_detection_report

    typer.echo(f"[detect-events] running sim-to-real experiment -> {output}")
    result = run_experiment(n_domains=n_domains, n_splits=n_splits, seed=seed)
    build_detection_report(result, output)
    tstr = result.regimes.get("TSTR")
    trtr = result.regimes.get("TRTR")
    if tstr and trtr:
        typer.echo(
            f"[detect-events] TSTR macroF1={tstr.f1_macro:.3f} "
            f"TRTR macroF1={trtr.f1_macro:.3f} -> {output}/detection_report.md"
        )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
