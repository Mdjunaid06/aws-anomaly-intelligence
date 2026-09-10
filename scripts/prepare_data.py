"""Prepare canonical processed observations from immutable NOAA source files."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.preprocessing import prepare_observations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    observations = prepare_observations()
    print(f"Processed rows: {len(observations)}")
    print(f"Stations: {observations['station_id'].nunique()}")