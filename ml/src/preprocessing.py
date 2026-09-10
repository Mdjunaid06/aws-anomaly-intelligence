"""Observation validation and canonical processed-table construction."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.paths import PROCESSED_OBSERVATIONS, RAW_DIR, STATION_METADATA, ensure_dirs

LOGGER = logging.getLogger(__name__)

START_DATE = "2024-01-01"
END_DATE = "2026-01-01"

VALUE_COLUMNS = [
    "temperature",
    "relative_humidity",
    "station_level_pressure",
    "sea_level_pressure",
]


def _suspect_quality(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip().str.lower()
    return (
        (numeric.notna() & ~numeric.isin([1.0, 1]))
        | as_str.isin(["f", "s", "x", "2", "3", "4", "5", "6", "7", "8", "9"])
    )


def process_station(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="|", low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[(df["DATE"] >= START_DATE) & (df["DATE"] < END_DATE)].copy()
    if df.empty:
        return df

    for column in VALUE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["pressure_hpa"] = df.get("station_level_pressure")
    df["pressure_type"] = "station_level"
    sea_level_mask = df["pressure_hpa"].isna() & df["sea_level_pressure"].notna()
    df.loc[sea_level_mask, "pressure_hpa"] = df.loc[sea_level_mask, "sea_level_pressure"]
    df.loc[sea_level_mask, "pressure_type"] = "sea_level"

    result = pd.DataFrame(
        {
            "timestamp": df["DATE"],
            "station_id": df["STATION"].astype(str),
            "station_name": df["Station_name"],
            "latitude": pd.to_numeric(df["LATITUDE"], errors="coerce"),
            "longitude": pd.to_numeric(df["LONGITUDE"], errors="coerce"),
            "elevation_m": pd.to_numeric(df["ELEVATION"], errors="coerce"),
            "temperature_c": df["temperature"],
            "relative_humidity_pct": df["relative_humidity"],
            "pressure_hpa": df["pressure_hpa"],
            "pressure_type": df["pressure_type"],
            "temperature_Quality_Code": df.get("temperature_Quality_Code"),
            "relative_humidity_Quality_Code": df.get("relative_humidity_Quality_Code"),
            "station_level_pressure_Quality_Code": df.get(
                "station_level_pressure_Quality_Code"
            ),
            "sea_level_pressure_Quality_Code": df.get("sea_level_pressure_Quality_Code"),
        }
    )
    return result


def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature_qc_suspect"] = _suspect_quality(df["temperature_Quality_Code"]).astype(int)
    df["humidity_qc_suspect"] = _suspect_quality(df["relative_humidity_Quality_Code"]).astype(int)
    pressure_qc = _suspect_quality(df["station_level_pressure_Quality_Code"])
    if "sea_level_pressure_Quality_Code" in df.columns:
        sea = _suspect_quality(df["sea_level_pressure_Quality_Code"])
        pressure_qc = pressure_qc | (
            df["pressure_type"].eq("sea_level") & sea & df["pressure_hpa"].notna()
        )
    df["pressure_qc_suspect"] = pressure_qc.astype(int)
    df["missing_temperature"] = df["temperature_c"].isna().astype(int)
    df["missing_humidity"] = df["relative_humidity_pct"].isna().astype(int)
    df["missing_pressure"] = df["pressure_hpa"].isna().astype(int)
    df["missing_any"] = (
        df["missing_temperature"] | df["missing_humidity"] | df["missing_pressure"]
    ).astype(int)
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"], keep="first"
    )
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("Dropped %s duplicate station_id+timestamp rows", dropped)
    return df.reset_index(drop=True)


def station_metadata_from_observations(df: pd.DataFrame, source_files: list[Path]) -> pd.DataFrame:
    lookup = {path.name.split("_")[1]: path.name for path in source_files if "_" in path.name}
    meta = (
        df.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            elevation_m=("elevation_m", "first"),
        )
        .copy()
    )
    meta["source_file"] = meta["station_id"].map(lambda sid: lookup.get(sid, ""))
    return meta


def prepare_observations(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Build processed observations from immutable raw NOAA PSV files."""
    _ = config or PipelineConfig()
    ensure_dirs()
    files = sorted(RAW_DIR.glob("GHCNh_*_por.psv"))
    if not files:
        raise FileNotFoundError(f"No NOAA PSV files found in {RAW_DIR}")

    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    for file_path in files:
        LOGGER.info("Processing %s", file_path.name)
        station_df = process_station(file_path)
        LOGGER.info("  rows in window: %s", len(station_df))
        if station_df.empty:
            raw = pd.read_csv(file_path, sep="|", nrows=1)
            metadata_rows.append(
                {
                    "station_id": str(raw.get("STATION", [file_path.stem])[0])
                    if "STATION" in raw.columns
                    else file_path.stem,
                    "station_name": raw["Station_name"].iloc[0] if "Station_name" in raw.columns else "",
                    "latitude": pd.to_numeric(raw["LATITUDE"], errors="coerce").iloc[0]
                    if "LATITUDE" in raw.columns
                    else None,
                    "longitude": pd.to_numeric(raw["LONGITUDE"], errors="coerce").iloc[0]
                    if "LONGITUDE" in raw.columns
                    else None,
                    "elevation_m": pd.to_numeric(raw["ELEVATION"], errors="coerce").iloc[0]
                    if "ELEVATION" in raw.columns
                    else None,
                    "source_file": file_path.name,
                }
            )
            continue
        frames.append(station_df)

    if not frames:
        raise RuntimeError("No observations found in the configured date window.")

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=False)
    combined = _drop_duplicates(combined)
    combined = _add_quality_flags(combined)
    combined = combined.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    combined.to_csv(PROCESSED_OBSERVATIONS, index=False)
    LOGGER.info("Wrote %s (%s rows)", PROCESSED_OBSERVATIONS, len(combined))

    meta = station_metadata_from_observations(combined, files)
    if metadata_rows:
        extra = pd.DataFrame(metadata_rows)
        meta = pd.concat([meta, extra], ignore_index=True)
        meta = meta.drop_duplicates(subset=["station_id"], keep="first")
    meta.to_csv(STATION_METADATA, index=False)
    return combined


def load_processed_observations(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_OBSERVATIONS
    if not path.exists():
        raise FileNotFoundError(f"Processed observations not found: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
