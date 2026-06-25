"""Loaders for reference / synthetic event data (spec section 7.1, 8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import EVENT_COLUMNS


def load_events(path: str | Path) -> pd.DataFrame:
    """Load an events CSV. Tolerant to column order; validates presence."""
    df = pd.read_csv(path)
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"events file {path} missing columns: {sorted(missing)}")
    return df[EVENT_COLUMNS]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
