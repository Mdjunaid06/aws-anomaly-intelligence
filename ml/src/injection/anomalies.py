"""Reproducible, validity-checked anomaly episodes from real observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.paths import GROUND_TRUTH_FILE, INJECTED_OBSERVATIONS, PROCESSED_OBSERVATIONS, ensure_dirs

VARIABLES = ("temperature_c", "relative_humidity_pct", "pressure_hpa")
EPISODE_TYPES = (
    "temperature_spike",
    "pressure_drop",
    "humidity_spike",
    "isolated_multivariate",
    "sensor_drift",
    "stuck_sensor",
    "missing_data",
    "communication_outage",
)


def _windows(df: pd.DataFrame, train_end: str, length: int) -> list[tuple[str, list[int]]]:
    windows: list[tuple[str, list[int]]] = []
    for station_id, group in df.groupby("station_id", sort=True):
        group = group.sort_values("timestamp")
        for start in range(max(len(group) - length + 1, 0)):
            indexes = group.index[start : start + length].tolist()
            times = group.loc[indexes, "timestamp"]
            if times.iloc[0] < pd.Timestamp(train_end):
                continue
            if (times.diff().dt.total_seconds().dropna() > 48 * 3600).any():
                continue
            if group.loc[indexes, list(VARIABLES)].isna().any().any():
                continue
            windows.append((str(station_id), indexes))
    return windows


def _choose(
    windows: list[tuple[str, list[int]]],
    used: set[tuple[str, int]],
    rng: np.random.Generator,
    predicate,
) -> tuple[str, list[int]] | None:
    for position in rng.permutation(len(windows)):
        station_id, indexes = windows[int(position)]
        if all((station_id, index) not in used for index in indexes) and predicate(station_id, indexes):
            return station_id, indexes
    return None


def _record(
    *, episode_id: str, anomaly_type: str, row: pd.Series, parameter: str,
    original: object, modified: object, offset: float | None,
    start_time: pd.Timestamp, end_time: pd.Timestamp, parameters: dict,
    seed: int,
) -> dict[str, object]:
    return {
        "anomaly_id": episode_id,
        "anomaly_type": anomaly_type,
        "station_id": str(row["station_id"]),
        "timestamp": pd.Timestamp(row["timestamp"]),
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "parameter": parameter,
        "affected_variable": parameter,
        "original_value": original,
        "modified_value": modified,
        "baseline_value": original,
        "injected_value": modified,
        "injected_offset": offset,
        "severity": abs(float(offset)) if offset is not None else 1.0,
        "injection_method": "seeded_controlled_episode",
        "scenario_parameters": json.dumps(parameters, sort_keys=True),
        "random_seed": seed,
        "validity_status": "valid_detector_test",
        "ground_truth_label": 1,
    }


def inject_anomalies(
    input_path: Path | None = None,
    output_path: Path | None = None,
    ground_truth_path: Path | None = None,
    config: PipelineConfig | None = None,
    episodes_per_type: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a separate injected copy and exact provenance labels."""
    config = config or PipelineConfig()
    ensure_dirs()
    source = Path(input_path or PROCESSED_OBSERVATIONS)
    output = Path(output_path or INJECTED_OBSERVATIONS)
    truth_path = Path(ground_truth_path or GROUND_TRUTH_FILE)
    if not source.exists():
        raise FileNotFoundError(f"Processed observations not found: {source}")

    df = pd.read_csv(source)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    rng = np.random.default_rng(config.random_seed)
    used: set[tuple[str, int]] = set()
    labels: list[dict[str, object]] = []
    episode_number = 0

    for anomaly_type in EPISODE_TYPES:
        length = 6 if anomaly_type == "sensor_drift" else 9 if anomaly_type == "stuck_sensor" else 3
        windows = _windows(df, config.train_end, length)
        for _ in range(episodes_per_type):
            def eligible(station_id: str, indexes: list[int]) -> bool:
                values = df.loc[indexes]
                if anomaly_type == "temperature_spike":
                    return float(values.temperature_c.iloc[0]) + 18.0 <= 48.0
                if anomaly_type == "pressure_drop":
                    return float(values.pressure_hpa.iloc[0]) - 35.0 >= 700.0
                if anomaly_type == "humidity_spike":
                    return float(values.relative_humidity_pct.iloc[0]) + 35.0 <= 95.0
                if anomaly_type == "isolated_multivariate":
                    return (
                        float(values.temperature_c.iloc[0]) + 12.0 <= 48.0
                        and float(values.relative_humidity_pct.iloc[0]) - 35.0 >= 5.0
                        and float(values.pressure_hpa.iloc[0]) + 20.0 <= 1090.0
                    )
                if anomaly_type == "stuck_sensor":
                    constant = float(values.relative_humidity_pct.iloc[0])
                    return 20.0 <= constant <= 85.0 and (values.relative_humidity_pct.iloc[1:] != constant).any()
                if anomaly_type == "sensor_drift":
                    offsets = np.linspace(1.0, 11.0, len(values))
                    modified = values.temperature_c.to_numpy(dtype=float) + offsets
                    return bool(np.all(np.diff(modified) > 0) and modified.max() <= 48.0)
                return True

            selected = _choose(windows, used, rng, eligible)
            if selected is None:
                continue
            station_id, indexes = selected
            used.update((station_id, index) for index in indexes)
            episode_number += 1
            episode_id = f"{anomaly_type}_{episode_number:03d}_{station_id}"
            start_time = pd.Timestamp(df.loc[indexes[0], "timestamp"])
            end_time = pd.Timestamp(df.loc[indexes[-1], "timestamp"])
            parameters = {"window_length": length, "scenario": anomaly_type}

            if anomaly_type == "temperature_spike":
                index = indexes[0]
                original = float(df.at[index, "temperature_c"])
                modified = original + 18.0
                df.at[index, "temperature_c"] = modified
                labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="temperature_c", original=original, modified=modified, offset=18.0, start_time=start_time, end_time=start_time, parameters=parameters, seed=config.random_seed))
            elif anomaly_type == "pressure_drop":
                index = indexes[0]
                original = float(df.at[index, "pressure_hpa"])
                modified = original - 35.0
                df.at[index, "pressure_hpa"] = modified
                labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="pressure_hpa", original=original, modified=modified, offset=-35.0, start_time=start_time, end_time=start_time, parameters=parameters, seed=config.random_seed))
            elif anomaly_type == "humidity_spike":
                index = indexes[0]
                original = float(df.at[index, "relative_humidity_pct"])
                modified = original + 35.0
                df.at[index, "relative_humidity_pct"] = modified
                labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="relative_humidity_pct", original=original, modified=modified, offset=35.0, start_time=start_time, end_time=start_time, parameters=parameters, seed=config.random_seed))
            elif anomaly_type == "isolated_multivariate":
                changes = {"temperature_c": 12.0, "relative_humidity_pct": -35.0, "pressure_hpa": 20.0}
                index = indexes[0]
                for parameter, offset in changes.items():
                    original = float(df.at[index, parameter])
                    modified = original + offset
                    df.at[index, parameter] = modified
                    labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter=parameter, original=original, modified=modified, offset=offset, start_time=start_time, end_time=start_time, parameters=parameters, seed=config.random_seed))
            elif anomaly_type == "sensor_drift":
                offsets = np.linspace(1.0, 6.0, len(indexes))
                for index, offset in zip(indexes, offsets):
                    original = float(df.at[index, "temperature_c"])
                    modified = original + float(offset)
                    df.at[index, "temperature_c"] = modified
                    labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="temperature_c", original=original, modified=modified, offset=float(offset), start_time=start_time, end_time=end_time, parameters={**parameters, "offsets": offsets.tolist()}, seed=config.random_seed))
            elif anomaly_type == "stuck_sensor":
                constant = float(df.at[indexes[0], "relative_humidity_pct"])
                for index in indexes:
                    original = float(df.at[index, "relative_humidity_pct"])
                    df.at[index, "relative_humidity_pct"] = constant
                    if original != constant:
                        labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="relative_humidity_pct", original=original, modified=constant, offset=constant - original, start_time=start_time, end_time=end_time, parameters={**parameters, "constant_value": constant}, seed=config.random_seed))
            elif anomaly_type == "missing_data":
                for index in indexes:
                    original = float(df.at[index, "relative_humidity_pct"])
                    df.at[index, "relative_humidity_pct"] = np.nan
                    labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter="relative_humidity_pct", original=original, modified=np.nan, offset=None, start_time=start_time, end_time=end_time, parameters={**parameters, "missing_variable": "relative_humidity_pct"}, seed=config.random_seed))
            elif anomaly_type == "communication_outage":
                for index in indexes:
                    for parameter in VARIABLES:
                        original = float(df.at[index, parameter])
                        df.at[index, parameter] = np.nan
                        labels.append(_record(episode_id=episode_id, anomaly_type=anomaly_type, row=df.loc[index], parameter=parameter, original=original, modified=np.nan, offset=None, start_time=start_time, end_time=end_time, parameters={**parameters, "missing_variables": list(VARIABLES)}, seed=config.random_seed))

    result = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    truth = pd.DataFrame(labels)
    if truth.empty:
        raise RuntimeError("No eligible anomaly windows were available.")
    result.to_csv(output, index=False)
    truth.to_csv(truth_path, index=False)
    return result, truth


if __name__ == "__main__":
    injected, labels = inject_anomalies()
    print(f"Saved {len(injected)} observations to {INJECTED_OBSERVATIONS}")
    print(f"Saved {len(labels)} labels in {labels['anomaly_id'].nunique()} episodes")
