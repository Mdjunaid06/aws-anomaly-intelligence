"""Nearby-station matching and time-aligned spatial evidence features."""

from __future__ import annotations

import logging
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.paths import STATION_METADATA
from ml.src.spatial_reasoning import SpatialConfig, StationSignal, calculate_spatial_evidence

LOGGER = logging.getLogger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    angle = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(min(1.0, angle)))


def load_station_metadata(path=None) -> pd.DataFrame:
    path = path or STATION_METADATA
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}")
    return pd.read_csv(path)


def neighbor_table(metadata: pd.DataFrame, max_distance_km: float) -> pd.DataFrame:
    rows: list[dict] = []
    stations = metadata.dropna(subset=["latitude", "longitude"]).to_dict("records")
    for target in stations:
        for other in stations:
            if target["station_id"] == other["station_id"]:
                continue
            distance = haversine_km(
                float(target["latitude"]),
                float(target["longitude"]),
                float(other["latitude"]),
                float(other["longitude"]),
            )
            if distance <= max_distance_km:
                rows.append(
                    {
                        "station_id": target["station_id"],
                        "neighbor_id": other["station_id"],
                        "distance_km": distance,
                    }
                )
    table = pd.DataFrame(rows)
    LOGGER.info("Neighbor pairs within %.0f km: %s", max_distance_km, len(table))
    return table


def _recent_sequence(frame: pd.DataFrame, timestamp, column: str) -> tuple[float | None, ...]:
    past = frame.loc[frame["timestamp"] <= timestamp, column].tail(3)
    return tuple(None if pd.isna(value) else float(value) for value in past.tolist())


def attach_spatial_features(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Add per-variable spatial evidence using time-aligned neighbors."""
    config = config or PipelineConfig()
    metadata = metadata if metadata is not None else load_station_metadata()
    neighbors = neighbor_table(metadata, config.max_neighbor_distance_km)
    prefixes = (
        ("temperature", "temperature_c"),
        ("humidity", "relative_humidity_pct"),
        ("pressure", "pressure_hpa"),
    )
    if neighbors.empty:
        LOGGER.warning("No neighbors within %.1f km", config.max_neighbor_distance_km)
        df = df.copy()
        for prefix, _ in prefixes:
            df[f"{prefix}_spatial_score"] = np.nan
            df[f"{prefix}_spatial_agreement"] = np.nan
            df[f"{prefix}_spatial_disagreement"] = np.nan
            df[f"{prefix}_usable_neighbors"] = 0
            df[f"{prefix}_missing_neighbors"] = 0
            df[f"{prefix}_insufficient_spatial"] = 1
        df["spatial_score"] = np.nan
        df["spatial_disagreement"] = np.nan
        df["spatial_common_mode_risk"] = 0.0
        df["insufficient_spatial"] = 1
        return df

    elevation = metadata.set_index("station_id")["elevation_m"].to_dict()
    tolerance = pd.Timedelta(config.spatial_tolerance)
    spatial_cfg = SpatialConfig(
        max_neighbor_distance_km=config.max_neighbor_distance_km,
        distance_scale_km=config.distance_scale_km,
        temporal_tolerance_minutes=max(tolerance.total_seconds() / 60.0, 1.0),
    )
    df = df.sort_values(["station_id", "timestamp"]).copy()
    by_station = {sid: group.sort_values("timestamp") for sid, group in df.groupby("station_id")}
    expected = neighbors.groupby("station_id")["neighbor_id"].nunique().to_dict()
    records: dict[tuple, dict] = {}

    for prefix, variable in prefixes:
        for station_id, target in by_station.items():
            station_neighbors = neighbors.loc[neighbors["station_id"] == station_id]
            total = int(expected.get(station_id, 0))
            aligned_frames: list[pd.DataFrame] = []
            for row in station_neighbors.itertuples(index=False):
                neighbor_df = by_station.get(row.neighbor_id)
                if neighbor_df is None:
                    continue
                left = target[["timestamp", variable, f"{prefix}_change"]].sort_values("timestamp")
                right = neighbor_df[["timestamp", variable, f"{prefix}_change"]].rename(
                    columns={
                        "timestamp": "neighbor_timestamp",
                        variable: "neighbor_value",
                        f"{prefix}_change": "neighbor_change",
                    }
                ).sort_values("neighbor_timestamp")
                merged = pd.merge_asof(
                    left,
                    right,
                    left_on="timestamp",
                    right_on="neighbor_timestamp",
                    direction="backward",
                    tolerance=tolerance,
                )
                merged["neighbor_id"] = row.neighbor_id
                merged["distance_km"] = row.distance_km
                aligned_frames.append(merged)

            if not aligned_frames:
                for timestamp in target["timestamp"]:
                    key = (station_id, pd.Timestamp(ts := timestamp))
                    records.setdefault(key, {})
                    records[key].update(_empty_spatial(prefix, total))
                continue

            aligned = pd.concat(aligned_frames, ignore_index=True)
            for timestamp, slice_df in aligned.groupby("timestamp"):
                target_rows = target.loc[target["timestamp"] == timestamp]
                if target_rows.empty:
                    continue
                target_value = target_rows[variable].iloc[0]
                target_change = target_rows[f"{prefix}_change"].iloc[0]
                baseline = (
                    None
                    if pd.isna(target_value) or pd.isna(target_change)
                    else float(target_value - target_change)
                )
                current = None if pd.isna(target_value) else float(target_value)
                target_signal = StationSignal(
                    station_id=station_id,
                    baseline_value=baseline,
                    current_value=current,
                    distance_km=0.0,
                    reliability=1.0,
                    data_quality=0.0 if pd.isna(target_value) else 1.0,
                    elevation_m=elevation.get(station_id),
                    sequence=_recent_sequence(target, timestamp, variable),
                )
                neighbor_signals = []
                for item in slice_df.itertuples(index=False):
                    usable = pd.notna(item.neighbor_value) and pd.notna(item.neighbor_change)
                    neighbor_df = by_station.get(item.neighbor_id)
                    neighbor_signals.append(
                        StationSignal(
                            station_id=item.neighbor_id,
                            baseline_value=(
                                None if not usable else float(item.neighbor_value) - float(item.neighbor_change)
                            ),
                            current_value=None if pd.isna(item.neighbor_value) else float(item.neighbor_value),
                            distance_km=float(item.distance_km),
                            reliability=1.0,
                            data_quality=1.0 if usable else 0.0,
                            elevation_m=elevation.get(item.neighbor_id),
                            sequence=_recent_sequence(neighbor_df, timestamp, variable)
                            if neighbor_df is not None
                            else (),
                        )
                    )
                evidence = calculate_spatial_evidence(
                    target_signal,
                    neighbor_signals,
                    total_neighbors=total,
                    config=spatial_cfg,
                )
                disagreement = (
                    (1.0 - evidence.agreement_score) * evidence.coverage_score
                    if not evidence.insufficient_coverage
                    else np.nan
                )
                key = (station_id, pd.Timestamp(timestamp))
                records.setdefault(key, {})
                records[key].update(
                    {
                        f"{prefix}_spatial_score": evidence.spatial_score,
                        f"{prefix}_spatial_agreement": evidence.agreement_score,
                        f"{prefix}_spatial_coverage": evidence.coverage_score,
                        f"{prefix}_spatial_disagreement": disagreement,
                        f"{prefix}_usable_neighbors": evidence.usable_neighbors,
                        f"{prefix}_missing_neighbors": evidence.missing_neighbors,
                        f"{prefix}_insufficient_spatial": int(evidence.insufficient_coverage),
                        f"{prefix}_common_mode_risk": evidence.common_mode_risk,
                        f"{prefix}_temporal_alignment_score": evidence.temporal_alignment_score,
                        f"{prefix}_elevation_context_score": evidence.elevation_context_score,
                        f"{prefix}_magnitude_similarity_score": evidence.magnitude_similarity_score,
                        f"{prefix}_direction_similarity_score": evidence.direction_similarity_score,
                        f"{prefix}_station_reliability": evidence.station_reliability,
                        f"{prefix}_total_neighbors": evidence.total_neighbors,
                        f"{prefix}_agreeing_stations": "|".join(evidence.agreeing_stations),
                        f"{prefix}_contradicting_stations": "|".join(evidence.contradicting_stations),
                    }
                )

    spatial_df = pd.DataFrame(
        [{"station_id": sid, "timestamp": ts, **values} for (sid, ts), values in records.items()]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if spatial_df.empty:
        df["spatial_score"] = np.nan
        df["spatial_disagreement"] = np.nan
        df["spatial_common_mode_risk"] = 0.0
        df["insufficient_spatial"] = 1
        return df

    spatial_df["timestamp"] = pd.to_datetime(spatial_df["timestamp"])
    df = df.merge(spatial_df, on=["station_id", "timestamp"], how="left")
    score_cols = [f"{prefix}_spatial_score" for prefix, _ in prefixes]
    disagreement_cols = [f"{prefix}_spatial_disagreement" for prefix, _ in prefixes]
    df["spatial_score"] = df[score_cols].max(axis=1, skipna=True)
    df["spatial_disagreement"] = df[disagreement_cols].max(axis=1, skipna=True)
    risk_cols = [c for c in df.columns if c.endswith("_common_mode_risk")]
    df["spatial_common_mode_risk"] = df[risk_cols].max(axis=1, skipna=True) if risk_cols else 0.0
    insuff_cols = [c for c in df.columns if c.endswith("_insufficient_spatial")]
    df["insufficient_spatial"] = df[insuff_cols].max(axis=1).fillna(1).astype(int)
    LOGGER.info("Attached spatial features for %s rows", len(df))
    return df


def _empty_spatial(prefix: str, total: int) -> dict:
    return {
        f"{prefix}_spatial_score": 0.0,
        f"{prefix}_spatial_agreement": 0.0,
        f"{prefix}_spatial_coverage": 0.0,
        f"{prefix}_spatial_disagreement": np.nan,
        f"{prefix}_usable_neighbors": 0,
        f"{prefix}_missing_neighbors": total,
        f"{prefix}_insufficient_spatial": 1,
        f"{prefix}_common_mode_risk": 0.0,
        f"{prefix}_temporal_alignment_score": 0.0,
        f"{prefix}_elevation_context_score": 0.0,
        f"{prefix}_magnitude_similarity_score": 0.0,
        f"{prefix}_direction_similarity_score": 0.0,
        f"{prefix}_station_reliability": 0.0,
        f"{prefix}_total_neighbors": total,
    }
