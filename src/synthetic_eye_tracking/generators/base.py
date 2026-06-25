"""Abstract base class shared by rule-based and model-based generators.

Both generators expose the same surface so the CLI and evaluation treat them
interchangeably (spec section 14: same schema out).
"""

from __future__ import annotations

import abc

import pandas as pd

from ..config import Config


class BaseGenerator(abc.ABC):
    """Common interface for all generators."""

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def generate_events(self, *args, **kwargs) -> pd.DataFrame:
        """Return an events DataFrame conforming to ``schema.EVENT_COLUMNS``."""

    @abc.abstractmethod
    def generate_raw_gaze(self, events: pd.DataFrame) -> pd.DataFrame:
        """Expand events into a raw gaze time-series (``schema.RAW_GAZE_COLUMNS``)."""

    @abc.abstractmethod
    def generate_aoi_summary(self, events: pd.DataFrame) -> pd.DataFrame:
        """Aggregate events into AOI-level summary (``schema.AOI_SUMMARY_COLUMNS``)."""
