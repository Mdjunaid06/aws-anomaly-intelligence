"""Train and run the explainable multi-evidence anomaly pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.detectors.isolation_forest import IsolationForestDetector
from ml.src.detectors.multivariate import MahalanobisDetector
from ml.src.detectors.rule_qc import rule_qc_scores
from ml.src.detectors.temporal import temporal_scores
from ml.src.evidence_fusion import EvidenceInput, FusionConfig, fuse_evidence
from ml.src.features.build_features import build_feature_frame
from ml.src.features.spatial_features import attach_spatial_features
from ml.src.models.health import add_sensor_health
from ml.src.preprocessing import load_processed_observations

LOGGER = logging.getLogger(__name__)


def _prepare(observations: pd.DataFrame, config: PipelineConfig, spatial: bool) -> pd.DataFrame:
    features = build_feature_frame(observations, config)
    features = temporal_scores(features)
    features = rule_qc_scores(features, config)
    if spatial:
        features = attach_spatial_features(features, config=config)
    else:
        features["spatial_score"] = 0.0
        features["spatial_disagreement"] = np.nan
        features["spatial_common_mode_risk"] = 0.0
        features["insufficient_spatial"] = 1
    return features


def fit_models(
    observations: pd.DataFrame,
    config: PipelineConfig | None = None,
    model_dir: Path | None = None,
) -> dict[str, object]:
    """Fit detectors only on observations before the configured train boundary."""
    config = config or PipelineConfig()
    features = _prepare(observations, config, spatial=False)
    train = features[features["timestamp"] < pd.Timestamp(config.train_end)].copy()
    if train.empty:
        raise RuntimeError("Training window contains no observations.")
    isolation = IsolationForestDetector(config).fit(train)
    mahalanobis = MahalanobisDetector(config).fit(train)
    isolation.save(model_dir)
    mahalanobis.save(model_dir)
    LOGGER.info("Fitted detectors on %s rows before %s", len(train), config.train_end)
    return {"isolation_forest": isolation, "mahalanobis": mahalanobis, "train_rows": len(train)}


def _spatial_evidence(row: pd.Series, prefix: str):
    from ml.src.spatial_reasoning import SpatialEvidence

    if pd.isna(row.get(f"{prefix}_spatial_score")):
        return None
    def names(column: str) -> tuple[str, ...]:
        value = row.get(column, "")
        return tuple(str(value).split("|")) if value and str(value) != "nan" else ()
    usable = int(row.get(f"{prefix}_usable_neighbors", 0))
    missing = int(row.get(f"{prefix}_missing_neighbors", 0))
    total = int(row.get(f"{prefix}_total_neighbors", usable + missing))
    return SpatialEvidence(
        spatial_score=float(row[f"{prefix}_spatial_score"]),
        agreement_score=float(row.get(f"{prefix}_spatial_agreement", 0.0)),
        coverage_score=float(row.get(f"{prefix}_spatial_coverage", 0.0)),
        temporal_alignment_score=float(row.get(f"{prefix}_temporal_alignment_score", 0.0)),
        elevation_context_score=float(row.get(f"{prefix}_elevation_context_score", 0.0)),
        magnitude_similarity_score=float(row.get(f"{prefix}_magnitude_similarity_score", 0.0)),
        direction_similarity_score=float(row.get(f"{prefix}_direction_similarity_score", 0.0)),
        station_reliability=float(row.get(f"{prefix}_station_reliability", 0.0)),
        common_mode_risk=float(row.get(f"{prefix}_common_mode_risk", 0.0)),
        total_neighbors=total,
        usable_neighbors=usable,
        missing_neighbors=missing,
        agreeing_stations=names(f"{prefix}_agreeing_stations"),
        contradicting_stations=names(f"{prefix}_contradicting_stations"),
        geographically_coherent=len(names(f"{prefix}_agreeing_stations")) >= 2,
        insufficient_coverage=bool(row.get(f"{prefix}_insufficient_spatial", 1)),
    )


def score_observations(
    observations: pd.DataFrame,
    models: dict[str, object],
    config: PipelineConfig | None = None,
    include_spatial: bool = True,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    features = _prepare(observations, config, spatial=include_spatial)
    isolation = models["isolation_forest"]
    mahalanobis = models["mahalanobis"]
    features["isolation_score"] = isolation.score(features)
    features["multivariate_score"] = mahalanobis.score(features)
    decisions = []
    for _, row in features.iterrows():
        spatial_by_variable = {
            prefix: evidence
            for prefix in ("temperature", "humidity", "pressure")
            if (evidence := _spatial_evidence(row, prefix)) is not None
        } if include_spatial else {}
        spatial = (
            max(spatial_by_variable.values(), key=lambda value: value.spatial_score)
            if spatial_by_variable else None
        )
        evidence = EvidenceInput(
            temporal_score=float(row.get("temporal_score", 0.0) or 0.0),
            multivariate_score=float(row.get("multivariate_score", 0.0) or 0.0),
            isolation_score=float(row.get("isolation_score", 0.0) or 0.0),
            rule_qc_score=float(row.get("rule_qc_score", 0.0) or 0.0),
            data_quality_score=1.0 - float(row.get("missing_any", 0) or 0),
            persistence_score=float(max(
                row.get("temperature_flatline_run", 0) or 0,
                row.get("humidity_flatline_run", 0) or 0,
                row.get("pressure_flatline_run", 0) or 0,
            )) / 4.0,
            spatial=spatial,
            spatial_by_variable=spatial_by_variable,
            drift_score=float(row.get("drift_score", 0.0) or 0.0),
            stuck_score=float(row.get("stuck_score", 0.0) or 0.0),
            communication_score=float(row.get("communication_score", 0.0) or 0.0),
        )
        decisions.append(fuse_evidence(evidence, target_station_id=str(row["station_id"])))
    result = pd.DataFrame([decision.as_dict() for decision in decisions])
    result = pd.concat([features.reset_index(drop=True), result], axis=1)
    return add_sensor_health(result)


def load_and_score(
    path: Path, models: dict[str, object], config: PipelineConfig | None = None,
) -> pd.DataFrame:
    return score_observations(load_processed_observations(path), models, config)