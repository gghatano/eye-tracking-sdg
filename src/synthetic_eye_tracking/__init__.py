"""Synthetic eye-tracking data generation package.

Two parallel generators (rule-based and model-based) emit data conforming to a
shared schema, evaluated with a single comparison framework.
"""

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
