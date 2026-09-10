"""Centralized, tunable pipeline configuration. No scattered magic numbers."""

from __future__ import annotations

from dataclasses import dataclass, field


RANDOM_SEED = 42


@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int = RANDOM_SEED
    train_end: str = "2025-01-01"

    # Physical / regional QC bounds for the Pune AWS network.
    temperature_min_c: float = -10.0
    temperature_max_c: float = 50.0
    humidity_min_pct: float = 0.0
    humidity_max_pct: float = 100.0
    pressure_min_hpa: float = 700.0
    pressure_max_hpa: float = 1100.0
    temperature_rate_limit_c_per_h: float = 12.0
    humidity_rate_limit_pct_per_h: float = 40.0
    pressure_rate_limit_hpa_per_h: float = 8.0

    # Time-aware feature windows. These stations are synoptic, not 5-minute AWS.
    rolling_window: str = "7D"
    rolling_min_periods: int = 3
    persistence_window: str = "24h"
    communication_gap_hours: float = 48.0
    spatial_tolerance: str = "3h"
    max_neighbor_distance_km: float = 250.0
    distance_scale_km: float = 100.0

    isolation_contamination: float = 0.02
    isolation_estimators: int = 200
    mahalanobis_flag_percentile: float = 99.0

    variables: tuple[str, ...] = field(
        default_factory=lambda: ("temperature_c", "relative_humidity_pct", "pressure_hpa")
    )
