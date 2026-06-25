"""Writers for the standard output bundle (spec section 10)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_generation_bundle(
    output_dir: str | Path,
    *,
    events: pd.DataFrame,
    raw_gaze: pd.DataFrame,
    aoi_summary: pd.DataFrame,
    participants: pd.DataFrame | None = None,
    stimuli: pd.DataFrame | None = None,
    aoi_layout: pd.DataFrame | None = None,
    config: Config | None = None,
) -> dict[str, Path]:
    """Write the full generation bundle to ``output_dir``.

    Returns a mapping of logical name -> written path. Plots are written
    separately by the evaluation module.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["events"] = _write_csv(events, out / "events.csv")
    written["raw_gaze"] = _write_csv(raw_gaze, out / "raw_gaze.csv")
    written["aoi_summary"] = _write_csv(aoi_summary, out / "aoi_summary.csv")
    if participants is not None:
        written["participants"] = _write_csv(participants, out / "participants.csv")
    if stimuli is not None:
        written["stimuli"] = _write_csv(stimuli, out / "stimuli.csv")
    if aoi_layout is not None:
        written["aoi_layout"] = _write_csv(aoi_layout, out / "aoi_layout.csv")
    if config is not None:
        config.to_yaml(out / "generation_config.yaml")
        written["generation_config"] = out / "generation_config.yaml"

    return written


def write_json(obj: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return p


def write_text(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
