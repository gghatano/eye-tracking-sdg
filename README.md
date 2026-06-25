# Synthetic Eye-Tracking Data Generator

Synthetic eye-tracking data for a **reading task**, produced by two parallel
generators and compared with a single evaluation framework:

- **Rule-based generator** — synthesizes gaze behavior from explicit
  assumptions, distributions, and transition rules.
- **Model-based generator** — learns statistical parameters / transition
  probabilities from (pseudo-)reference data, then samples.

Both emit the **same schema** (event-level, raw gaze time-series, AOI summary),
so quality can be compared on identical metrics.

## Status

🚧 Under active development. Milestone tracking:

- **M1** — rule-based generator, reading-grid AOI layout, fixation/saccade/blink
  events, 120 Hz raw gaze expansion, basic evaluation report.
- **M2** — pseudo-reference fitting, parametric + Markov model-based generator,
  rule-based vs model-based comparison.
- **M3** — public dataset adaptation, downstream utility, privacy-proxy eval.

## Quickstart (uv)

```bash
uv sync

# Rule-based generation
uv run synthetic-eye-tracking generate-rule \
  --config configs/rule_based.yaml \
  --output data/synthetic/rule_based/
```

See the spec in this repository for full details.

## Development

```bash
uv sync
uv run pytest
```
