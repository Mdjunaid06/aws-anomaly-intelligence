"""Deterministic meteorological quality-control scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig


def rule_qc_scores(df: pd.DataFrame, config: PipelineConfig | None = None) -> pd.DataFrame:
    config = config or PipelineConfig()
    out = df.copy()
    t = out["temperature_c"]
    h = out["relative_humidity_pct"]
    p = out["pressure_hpa"]

    range_t = ((t < config.temperature_min_c) | (t > config.temperature_max_c)) & t.notna()
    range_h = ((h < config.humidity_min_pct) | (h > config.humidity_max_pct)) & h.notna()
    range_p = ((p < config.pressure_min_hpa) | (p > config.pressure_max_hpa)) & p.notna()
    rate_t = out["temperature_rate"].abs() > config.temperature_rate_limit_c_per_h
    rate_h = out["humidity_rate"].abs() > config.humidity_rate_limit_pct_per_h
    rate_p = out["pressure_rate"].abs() > config.pressure_rate_limit_hpa_per_h
    temperature_stuck = (
        (out["temperature_flatline_run"].fillna(0) >= 2)
        & (out["temperature_persistence_hours"].fillna(0) >= 24)
    )
    humidity_stuck = (
        (out["humidity_flatline_run"].fillna(0) >= 2)
        & (out["humidity_persistence_hours"].fillna(0) >= 24)
        & (h < config.humidity_max_pct)
    )
    pressure_stuck = (
        (out["pressure_flatline_run"].fillna(0) >= 2)
        & (out["pressure_persistence_hours"].fillna(0) >= 24)
    )
    stuck = temperature_stuck | humidity_stuck | pressure_stuck
    gap = out.get("source_sampling_gap", pd.Series(0, index=out.index)).fillna(0).astype(bool)
    missing = out["missing_any"].astype(bool) if "missing_any" in out.columns else t.isna() | h.isna() | p.isna()
    qc_suspect = np.zeros(len(out), dtype=bool)
    for column in ("temperature_qc_suspect", "humidity_qc_suspect", "pressure_qc_suspect"):
        if column in out.columns:
            qc_suspect = qc_suspect | out[column].fillna(0).astype(bool).to_numpy()

    flags = pd.DataFrame(
        {
            "qc_range": (range_t | range_h | range_p).astype(float),
            "qc_rate": (rate_t | rate_h | rate_p).fillna(False).astype(float),
            "qc_stuck": stuck.astype(float),
            "qc_gap": gap.astype(float),
            "qc_missing": missing.astype(float),
            "qc_source_flag": qc_suspect.astype(float),
        },
        index=out.index,
    )
    out["rule_qc_score"] = flags.drop(columns=["qc_gap"]).max(axis=1)
    out["rule_range_flag"] = flags["qc_range"]
    out["rule_rate_flag"] = flags["qc_rate"]
    out["qc_stuck"] = flags["qc_stuck"]
    out["qc_gap"] = flags["qc_gap"]
    out["communication_score"] = np.clip(
        out.get("all_variables_missing", pd.Series(0, index=out.index)).astype(float) * 0.8,
        0,
        1,
    )
    out["temperature_stuck_score"] = temperature_stuck.astype(float)
    out["humidity_stuck_score"] = humidity_stuck.astype(float)
    out["pressure_stuck_score"] = pressure_stuck.astype(float)
    out["stuck_score"] = pd.concat(
        [out["temperature_stuck_score"], out["humidity_stuck_score"], out["pressure_stuck_score"]], axis=1
    ).max(axis=1)
    return out
