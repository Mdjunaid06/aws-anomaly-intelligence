"""Replay historical observations for a simple live-pipeline demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.paths import INJECTED_OBSERVATIONS, PROCESSED_OBSERVATIONS


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=Path, default=INJECTED_OBSERVATIONS)
	parser.add_argument("--limit", type=int, default=10)
	args = parser.parse_args()
	source = args.input if args.input.exists() else PROCESSED_OBSERVATIONS
	if not source.exists():
		raise FileNotFoundError(f"No replay input found at {source}")
	observations = pd.read_csv(source, parse_dates=["timestamp"])
	observations = observations.sort_values("timestamp").head(max(args.limit, 0))
	for row in observations.itertuples(index=False):
		print(f"{row.timestamp.isoformat()} station={row.station_id}")
	print(f"Replayed rows: {len(observations)}")


if __name__ == "__main__":
	main()
