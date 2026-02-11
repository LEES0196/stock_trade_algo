"""Strategy subpackage for the auto portfolio project.

This subpackage groups modules that handle data access, feature
engineering, allocation algorithms and blending logic.  See individual
modules for details.
"""

from .data import download_prices
from .features import monthly_returns
from .algo_rules import run_algorithm_1
from .algo_scores import run_algorithm_2
from .blend import blend
from .allocate import run_all

__all__ = [
    "download_prices",
    "monthly_returns",
    "run_algorithm_1",
    "run_algorithm_2",
    "blend",
    "run_all",
]