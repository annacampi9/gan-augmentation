"""Training configuration for DADA.

This module re-exports `DADAConfig` and `cfg` from the original `dada_config.py`
to minimize churn. Defaults and behavior remain unchanged.
"""

from dada_config import DADAConfig, cfg  # type: ignore

__all__ = ["DADAConfig", "cfg"]
