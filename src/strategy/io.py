"""Persistence helper functions for the auto portfolio strategy.

This module defines a helper for appending allocation decisions to a
CSV file.  Each row in the output file contains ``timestamp``, ``ticker``
and ``weight`` columns.  Calling ``append_decision`` repeatedly will
preserve existing rows and append new ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

def append_decision(df: pd.DataFrame, csv_path: str) -> None:
    """Append the contents of a decision DataFrame to a CSV file.

    The incoming DataFrame should have tickers in the index and at least
    ``timestamp`` and ``weight`` columns.  Any existing rows in the
    destination file are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index of tickers and columns ``timestamp`` and ``weight``.
    csv_path : str
        Path to the CSV file to append to.  Parent directories are created if
        they do not exist.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prepare the DataFrame for appending by resetting the index to a column
    prepared = df.reset_index().rename(columns={"index": "ticker"})
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            # If the file exists but is corrupt, start fresh
            old = pd.DataFrame(columns=["timestamp", "ticker", "weight"])
        out = pd.concat([old, prepared], ignore_index=True)
    else:
        out = prepared
    out.to_csv(path, index=False)