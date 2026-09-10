"""Create controlled anomalies from the real processed NOAA observations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.injection.anomalies import inject_anomalies


if __name__ == "__main__":
	injected, truth = inject_anomalies()
	print(f"Injected observations: {len(injected)}")
	print(f"Ground-truth rows: {len(truth)}")
