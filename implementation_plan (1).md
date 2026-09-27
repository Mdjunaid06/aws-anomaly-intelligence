# Implementation Plan - AWS Anomaly Intelligence: SIH 2026 Refinement & Completion

Transform the existing AWS Anomaly Intelligence prototype into an operations-grade weather observation quality platform for the Smart India Hackathon 2026. The intelligence engine (`ml/`) remains strictly frozen; all enhancements are delivered through backend API extensions, robust data synchronization, and a meteorological operations UI.

## User Review Required

> [!IMPORTANT]
> **Authoritative ML Pipeline Constraint**: In accordance with project architecture and SIH constraints, zero modifications will be made to `ml/`. All data adaptations, missing context handling, and evidence formatting will occur in the backend service layer and frontend presentation.
> 
> **Database Engine**: The application seamlessly supports SQLite (`aws_anomaly.db` in WAL mode for local zero-docker development and judging) as well as PostgreSQL. Both are verified via SQLAlchemy.

---

## Proposed Changes

### 1. Backend Service Layer & API Extensions

#### [MODIFY] [backend/app/schemas/replay.py](file:///d:/aws-anomaly-intelligence/backend/app/schemas/replay.py)
- Extend `ReplayStatus` to include `total_observations: int = 0` and `progress_pct: float = 0.0`.
- Support accurate backend-computed progress tracking for the Replay Lab.

#### [MODIFY] [backend/app/services/replay.py](file:///d:/aws-anomaly-intelligence/backend/app/services/replay.py)
- Compute `total_observations` on replay start and update `progress_pct` on each processed row.
- Ensure safe reset and speed transitions across all allowed speeds (`0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0`).

#### [MODIFY] [backend/app/services/batch.py](file:///d:/aws-anomaly-intelligence/backend/app/services/batch.py)
- In `_load_ml_context(db, obs)`: defensively filter out stations with no valid observation variables in the window before passing to `ml_engine.score_observation()`. This ensures that `ml/src/models/sequence/dataset.py` never encounters an empty valid group, resolving the `_segment` KeyError without modifying `ml/`.

#### [NEW] [backend/app/routes/system.py](file:///d:/aws-anomaly-intelligence/backend/app/routes/system.py)
- Add `GET /system/status`:
  - Returns real-time health checks for Database (`connected`, observation count, anomaly count), ML Pipeline (`online`, model directory, loaded detectors), Replay (`state`, processed count), and GenAI (`configured`, `provider`, `model`).

#### [MODIFY] [backend/app/routes/anomalies.py](file:///d:/aws-anomaly-intelligence/backend/app/routes/anomalies.py)
- Add `GET /anomalies/{pred_id}/spatial-context`:
  - Returns the target station's readings alongside synchronized observations from all neighbor stations within the 250km radius for the same observation window.
  - Supplies metadata for `insufficient_spatial_evidence` (usable neighbors, supporting stations, contradicting stations).
- Add detector availability metadata helper in `AnomalyEvidenceDetail` (distinguishing 0.0 vs unavailable due to missing temporal/sequence context).

#### [MODIFY] [backend/app/routes/stations.py](file:///d:/aws-anomaly-intelligence/backend/app/routes/stations.py)
- Add `GET /stations/{station_id}/history`:
  - Returns recent chronological weather observations (temperature, pressure, relative humidity) and anomaly flags for charting (6h, 24h, 7d, 30d).

#### [MODIFY] [backend/app/main.py](file:///d:/aws-anomaly-intelligence/backend/app/main.py)
- Register `system.router` into the FastAPI application.

---

### 2. Frontend Operations Console & Design Refinement

#### [MODIFY] [frontend/src/api.ts](file:///d:/aws-anomaly-intelligence/frontend/src/api.ts)
- Add typed API methods for `/system/status`, `/anomalies/{id}/spatial-context`, and `/stations/{id}/history`.
- Ensure strict TypeScript typing for spatial context, evidence availability, and system health.

#### [NEW] [frontend/src/components/SystemHeader.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/SystemHeader.tsx)
- Meteorological operations center top navigation bar.
- Live status indicators: PostgreSQL/SQLite DB state, ML Pipeline online, Replay state, and GenAI provider status.

#### [NEW] [frontend/src/components/NetworkOverview.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/NetworkOverview.tsx)
- Top operational KPIs (Stations monitored, Active anomalies, Stations on watch, Healthy stations, Total observations).
- High-precision Leaflet map with state-based markers (Operational green, Watch amber, Anomaly red).
- "Multi-signal decision (Why this is not just a threshold)" panel explaining evidence fusion to judges.
- "SIH Differentiation" panel highlighting spatiotemporal reasoning and confidence calibration.
- Recent alerts table with direct deep-link into Investigation.

#### [NEW] [frontend/src/components/StationDrawer.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/StationDrawer.tsx)
- Dedicated station detail panel with live sensor health, last observation values, and status.
- Recharts observation trend lines (Temperature °C, Pressure hPa, Relative Humidity %) with 6h/24h/7d/30d filter.
- Historical anomaly persistence timeline.

#### [NEW] [frontend/src/components/AnomalyInvestigation.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/AnomalyInvestigation.tsx)
- **Fixes Stale Explanation Bug**: Strictly isolates investigation state by `prediction_id`.
  - When alert changes, all stale state is wiped immediately and new record-specific explanation is retrieved.
- **Evidence Strength Horizontal Bars**: Visual bars for Temporal, Multivariate, Isolation Forest, Spatial, Rule QC, GRU, Persistence, Data Quality.
  - Gracefully displays "Not available" with contextual tooltip when detector prerequisites are not met.
- **Decision Confidence**: Labeled accurately with calibration disclaimer tooltip.
- **Insufficient Spatial Evidence**: Visual badge and explanation when neighboring stations are unavailable or unaligned.
- **Spatial Comparison Matrix**: Target station vs neighboring stations table showing weather readings around the same timestamp.
- **Decision Trace & Recommendations**: Operational next steps generated from ML evidence.
- **Collapsible Raw JSON**: Debug evidence kept accessible under an expandable toggle.

#### [NEW] [frontend/src/components/ReplayLab.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/ReplayLab.tsx)
- Historical Replay simulation console with live progress bar (`%` processed).
- Replay speed selector (`1x, 2x, 5x, 10x, 20x, 50x`).
- Replay controls: Start, Pause, Resume, Stop, Reset.
- Live incoming observation card with direct "Investigate Alert" action.

#### [NEW] [frontend/src/components/OperatorAssistant.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/components/OperatorAssistant.tsx)
- Focused operations Q&A with predefined operational prompts.
- Displays provenance, fallback status, and structured evidence backing every answer.

#### [MODIFY] [frontend/src/App.tsx](file:///d:/aws-anomaly-intelligence/frontend/src/App.tsx)
- Orchestrate navigation between Overview, Replay Lab, and Assistant with seamless station/anomaly drawer integration.

#### [MODIFY] [frontend/src/styles.css](file:///d:/aws-anomaly-intelligence/frontend/src/styles.css)
- Implement refined dark meteorological operations theme: crisp Inter/system typography, slate background, subtle 1px border dividers, high-contrast accessible status indicators, and clean data tables.

---

### 3. Verification & Testing

#### [NEW] [backend/tests/test_explanation_isolation.py](file:///d:/aws-anomaly-intelligence/backend/tests/test_explanation_isolation.py)
- Dedicated test explicitly proving:
  1. Open Anomaly A -> Fetch Explanation A.
  2. Open Anomaly B -> Fetch Explanation B.
  3. Assert Explanation B contains B's exact values (`station_id`, `timestamp`, `confidence`, `root_cause`, `evidence`).
  4. Assert Explanation B contains NO values from A.
- Automated API tests for new endpoints (`/system/status`, `/anomalies/{id}/spatial-context`, `/stations/{id}/history`).
- Run full pytest test suite (must pass 100%).
- Run `git diff --name-only -- ml` (must return zero output).
- Build frontend (`npm run build`) and test in live browser.
- Produce comprehensive final completion report with exact GenAI API key setup instructions.
