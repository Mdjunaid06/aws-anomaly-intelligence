"""ML engine adapter - wraps existing ML pipeline without duplication."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.src.config import PipelineConfig
from ml.src.evidence_fusion import FusionConfig, fuse_evidence
from ml.src.models.pipeline import fit_models, score_observations
from ml.src.models.health import add_sensor_health

from ..core.config import settings

LOGGER = logging.getLogger(__name__)


class MLEngine:
    """Adapter for existing ML pipeline.
    
    Non-negotiable rule: Do NOT duplicate ML calculations.
    This class only wraps existing ml/src functions.
    """

    def __init__(self):
        """Initialize ML engine with trained models."""
        self.config = PipelineConfig()
        self.fusion_config = FusionConfig()
        self.model_dir = settings.ml_models_dir
        self.models = self._load_models()

    def _load_models(self) -> dict:
        """Load pre-trained models from disk."""
        try:
            from ml.src.models.sequence import GRUDetector
            from ml.src.detectors.isolation_forest import IsolationForestDetector
            from ml.src.detectors.multivariate import MahalanobisDetector

            models = {
                "isolation_forest": IsolationForestDetector.load(self.model_dir),
                "mahalanobis": MahalanobisDetector.load(self.model_dir),
                "gru": GRUDetector.load(self.model_dir),
            }
            LOGGER.info("Loaded ML models from %s", self.model_dir)
            return models
        except Exception as e:
            LOGGER.error("Failed to load models: %s", e)
            raise

    def score_observation(
        self,
        observation_df: pd.DataFrame,
        include_spatial: bool = True,
    ) -> pd.DataFrame:
        """Score a single observation using existing pipeline.
        
        Args:
            observation_df: DataFrame with columns: station_id, timestamp, 
                           temperature_c, pressure_hpa, relative_humidity_pct
            include_spatial: Whether to include spatial reasoning
            
        Returns:
            DataFrame with anomaly scores and evidence
        """
        if observation_df.empty:
            LOGGER.warning("Empty observation DataFrame provided")
            return pd.DataFrame()

        try:
            # Call existing pipeline - NO DUPLICATION
            results = score_observations(
                observation_df,
                models=self.models,
                config=self.config,
                include_spatial=include_spatial,
            )
            LOGGER.debug("Scored %d observations", len(results))
            return results
        except Exception as e:
            LOGGER.error("Failed to score observations: %s", e)
            raise

    def score_batch(
        self,
        observations_df: pd.DataFrame,
        include_spatial: bool = True,
    ) -> pd.DataFrame:
        """Score multiple observations using existing pipeline.
        
        Args:
            observations_df: DataFrame with observation data
            include_spatial: Whether to include spatial reasoning
            
        Returns:
            DataFrame with anomaly scores for all observations
        """
        if observations_df.empty:
            LOGGER.warning("Empty batch provided")
            return pd.DataFrame()

        try:
            # Call existing pipeline - NO DUPLICATION
            results = score_observations(
                observations_df,
                models=self.models,
                config=self.config,
                include_spatial=include_spatial,
            )
            LOGGER.info("Scored batch of %d observations", len(results))
            return results
        except Exception as e:
            LOGGER.error("Failed to score batch: %s", e)
            raise

    def fuse_evidence(
        self,
        evidence_dict: dict,
        target_station_id: str,
    ) -> dict:
        """Fuse evidence into final anomaly decision using existing pipeline.
        
        Args:
            evidence_dict: Evidence from ML detectors
            target_station_id: Station ID for this observation
            
        Returns:
            FusedDecision as dictionary
        """
        try:
            from ml.src.evidence_fusion import EvidenceInput

            # Create EvidenceInput from detector output
            evidence = EvidenceInput(**evidence_dict)
            
            # Call existing fusion - NO DUPLICATION
            decision = fuse_evidence(
                evidence,
                target_station_id=target_station_id,
                config=self.fusion_config,
            )
            
            return decision.as_dict()
        except Exception as e:
            LOGGER.error("Failed to fuse evidence: %s", e)
            raise

    def get_model_version(self) -> str:
        """Get model artifact version/hash for provenance tracking."""
        try:
            # Try to read model file mtime or hash
            gru_path = self.model_dir / "gru_detector.joblib"
            if gru_path.exists():
                import hashlib
                with open(gru_path, 'rb') as f:
                    hash_obj = hashlib.md5(f.read())
                    return hash_obj.hexdigest()[:8]
        except Exception as e:
            LOGGER.warning("Could not compute model version: %s", e)
        return "unknown"


# Global ML engine instance
_ml_engine: Optional[MLEngine] = None


def get_ml_engine() -> MLEngine:
    """Get or initialize ML engine singleton."""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLEngine()
    return _ml_engine
