# AWS Anomaly Intelligence
# Engineering Implementation Handoff
# ML → Backend → PostgreSQL → GenAI → Frontend → Docker

> Project: SIH26073 — AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations
>
> Purpose: This document is the authoritative engineering handoff for continuing development after completion of the ML implementation.
>
> Audience: Human developers, teammates, and AI coding agents working inside this repository.
>
> IMPORTANT: Read this document completely before modifying backend, frontend, database, or integration code.

---

# 1. PURPOSE OF THIS DOCUMENT

The ML implementation is complete enough to act as the project's intelligence engine.

The next team members must build the application around the existing ML system.

The intended final flow is:

AWS / Historical Observation
        ↓
Data Validation
        ↓
Existing ML Pipeline
        ↓
Detector Evidence
        ↓
Evidence Fusion
        ↓
Anomaly Decision
        ↓
Root Cause
        ↓
Sensor Health
        ↓
PostgreSQL
        ↓
FastAPI
        ↓
React Frontend
        ↓
Operator Dashboard

Optional:

Structured ML Evidence
        ↓
GenAI Explanation
        ↓
Human-readable explanation

GenAI is NOT the anomaly detector.

The ML pipeline remains authoritative.

---

# 2. NON-NEGOTIABLE RULES

1. Do not rewrite the existing ML pipeline just to make backend integration easier.

2. Do not duplicate ML calculations in FastAPI.

3. Do not duplicate ML calculations in React.

4. Do not allow the frontend to decide whether an observation is anomalous.

5. Do not allow GenAI to override the ML decision.

6. Do not put GenAI API keys in frontend code.

7. Do not put database passwords in Git.

8. Do not modify raw NOAA observations.

9. Do not overwrite ground truth.

10. Keep injected anomalies separate from original observations.

11. Preserve station ID and timestamp throughout the complete pipeline.

12. Preserve provenance for every derived result.

13. Do not introduce Kafka, Spark, Kubernetes, or unnecessary microservices unless there is a demonstrated requirement.

14. Every backend change must continue passing the existing ML tests.

15. Every integration must be tested from the smallest layer upward.

---

# 3. CURRENT ML CHECKPOINT

Current committed ML checkpoint:

6e3735f

Commit:

Complete AWS anomaly detection ML pipeline

The ML implementation includes:

- Data preprocessing
- Feature engineering
- Rule-based QC
- Temporal detector
- Multivariate detector
- Isolation Forest
- Spatial reasoning
- Evidence fusion
- Root-cause classification
- Sensor health
- Controlled anomaly injection
- Ground truth generation
- Evaluation
- Replay
- GRU temporal detector

The ML implementation must now be treated as a stable component.

---

# 4. CURRENT DATASET

Current benchmark:

Clean observations:
6990

Training rows:
3463

Holdout rows:
3527

Injected episodes:
16

Ground-truth rows:
62

Working station count:
5

Required observation variables:

temperature_c
pressure_hpa
relative_humidity_pct

The project uses historical NOAA GHCNh data.

Raw data must remain separate from:

- processed data
- feature data
- injected data
- predictions
- ground truth
- application database records

---

# 5. CURRENT ML ARCHITECTURE

The existing conceptual architecture is:

                    AWS OBSERVATION
                          |
                          v
                    PREPROCESSING
                          |
             +------------+------------+
             |            |            |
             v            v            v
         TEMPORAL     MULTIVARIATE    RULE QC
            AI            AI
             |            |
             +------------+-------------+
                          |
                    ISOLATION FOREST
                          |
                    SPATIAL REASONING
                          |
                    GRU TEMPORAL
                          |
                          v
                   EVIDENCE FUSION
                          |
                          v
                   CONFIDENCE SCORE
                          |
                 +--------+--------+
                 |                 |
                 v                 v
             SENSOR FAULT      WEATHER EVENT
                 |                 |
                 +--------+--------+
                          |
                          v
                     ROOT CAUSE
                          |
                          v
                    SENSOR HEALTH
                          |
                          v
                     APPLICATION

Important:

GRU is an additional detector.

GRU does not replace the existing detectors.

---

# 6. GRU IMPLEMENTATION

GRU architecture:

Input:
24 observations × 3 variables

Variables:

temperature_c
relative_humidity_pct
pressure_hpa

Architecture:

GRU
- 1 layer
- hidden size 64

Task:

Predict the next weather observation.

Loss:

MSE prediction error.

Anomaly threshold:

99th percentile of training prediction error.

Output:

Normalized GRU anomaly score between 0 and 1.

Saved artifact:

ml/artifacts/models/gru_detector.joblib

Important GRU properties:

- Station-local
- Chronological
- Causal
- Missing-safe
- Gap-aware
- Does not cross station boundaries
- Trained only on clean training observations

The GRU currently does NOT improve the benchmark.

That must not be hidden or falsely represented.

---

# 7. CURRENT ML RESULTS

Existing baseline fusion:

Precision: 0.2308
Recall: 0.1304
F1: 0.1667
False Alarm Rate: 0.0057
Episodes: 6/16

GRU-only:

Precision: 0.0698
Recall: 0.0652
F1: 0.0674
False Alarm Rate: 0.0115
Episodes: 3/16

Fusion + GRU:

Precision: 0.2000
Recall: 0.1087
F1: 0.1408
False Alarm Rate: 0.0057
Episodes: 5/16

Conclusion:

GRU does not currently improve this sparse benchmark.

The correct engineering interpretation is:

GRU is available as an additional temporal evidence source, but the existing fusion pipeline currently performs better on this dataset.

Do not change thresholds simply to make GRU look better.

---

# 8. ML VALIDATION STATUS

Current ML validation:

pytest:

28 passed

Compile:

python -m compileall -q ml scripts backend

Passed.

Git validation:

git diff --check

Passed.

Before backend development, reproduce this state.

---

# 9. REPOSITORY STRUCTURE

Current major structure:

aws-anomaly-intelligence/

├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── docs/
│   ├── 01-PROJECT-BRAIN.md
│   ├── 02-PROBLEM-AND-SOLUTION.md
│   ├── 03-SYSTEM-ARCHITECTURE.md
│   ├── 04-DATA-COLLECTION-AND-VALIDATION.md
│   ├── 05-ML-PIPELINE-AND-EXPERIMENTS.md
│   ├── 06-APPLICATION-AND-API.md
│   ├── 07-DEMO-DEPLOYMENT-AND-TEAM-GUIDE.md
│   └── 08-IMPLEMENTATION-HANDOFF-ML-BACKEND-FRONTEND.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── injected/
│   └── ground_truth/
│
├── ml/
│   └── src/
│
├── scripts/
│
├── backend/
│   ├── app/
│   └── tests/
│
└── frontend/

---

# 10. STEP ZERO — REPRODUCE ML BEFORE CODING BACKEND

Every new developer must first prove that the repository works.

Windows:

git clone <repository-url>

cd aws-anomaly-intelligence

Create environment if required:

python -m venv .venv

Activate:

.venv\Scripts\activate

Install:

python -m pip install --upgrade pip

pip install -r requirements.txt

Run tests:

python -m pytest -q

Expected:

28 passed

Compile:

python -m compileall -q ml scripts backend

Then run:

python scripts/prepare_data.py

python scripts/train_model.py

python scripts/inject_anomalies.py

python scripts/evaluate_model.py

Do not begin backend implementation if the baseline ML environment does not work.

---

# 11. BACKEND ARCHITECTURE

The backend should be an orchestration and persistence layer.

It should NOT become another ML implementation.

Correct architecture:

                    FRONTEND
                       |
                       | HTTP / JSON
                       v
                    FASTAPI
                       |
                  Pydantic Schemas
                       |
                       v
                  Service Layer
                       |
             +---------+---------+
             |                   |
             v                   v
          ML ENGINE          PostgreSQL
             |
             v
       Existing ML Pipeline

GenAI:

Anomaly Evidence
       |
       v
GenAI Service
       |
       v
LLM Provider
       |
       v
Explanation

---

# 12. BACKEND FILE CREATION ORDER

Do not create everything randomly.

Implement in this order:

1. core/config.py
2. core/database.py
3. database models
4. Pydantic schemas
5. ML engine adapter
6. observation service
7. anomaly service
8. health service
9. replay service
10. API routes
11. GenAI service
12. GenAI route
13. tests
14. frontend
15. Docker
16. end-to-end validation

Each stage must work before moving to the next.

---

# 13. BACKEND CORE CONFIG

File:

backend/app/core/config.py

Responsibilities:

- Read environment variables.
- Database configuration.
- API host/port.
- CORS configuration.
- GenAI configuration.
- Environment name.
- ML artifact paths if necessary.

Use pydantic-settings.

Do not hardcode:

- database passwords
- API keys
- production URLs

Example environment:

DATABASE_URL=postgresql+psycopg2://aws_app:<password>@localhost:5432/aws_anomaly_intelligence

ENVIRONMENT=development

API_HOST=0.0.0.0

API_PORT=8000

FRONTEND_ORIGIN=http://localhost:5173

GENAI_API_KEY=<secret>

GENAI_MODEL=<provider-model>

---

# 14. POSTGRESQL SETUP

PostgreSQL is the application persistence layer.

Create database:

CREATE DATABASE aws_anomaly_intelligence;

Create application user:

CREATE USER aws_app WITH PASSWORD '<strong-password>';

Grant:

GRANT ALL PRIVILEGES ON DATABASE aws_anomaly_intelligence TO aws_app;

After connecting to the database:

GRANT USAGE, CREATE ON SCHEMA public TO aws_app;

Do not use the PostgreSQL superuser in the application.

Do not commit the password.

---

# 15. DATABASE DESIGN

Recommended tables:

stations

observations

anomalies

evidence

sensor_health

replay_jobs

Optional:

maintenance_events

explanations

audit_events

---

# 16. STATIONS TABLE

File:

backend/app/models/station.py

Purpose:

Store station identity and location.

Suggested fields:

id
station_id
name
latitude
longitude
active
created_at
updated_at

station_id must be unique.

Example:

INM00043067

The station ID must remain consistent with the ML dataset.

---

# 17. OBSERVATIONS TABLE

File:

backend/app/models/observation.py

Suggested fields:

id
station_id
timestamp
temperature_c
pressure_hpa
relative_humidity_pct
source
data_type
created_at

data_type examples:

raw
replayed
injected

Important:

Raw observations must never be overwritten.

A corrected/imputed value should be stored separately if implemented.

---

# 18. ANOMALIES TABLE

File:

backend/app/models/anomaly.py

Suggested fields:

id
observation_id
station_id
timestamp
anomaly
anomaly_score
confidence
classification
root_cause
created_at

This stores the final ML decision.

The backend must not calculate this decision itself.

---

# 19. EVIDENCE TABLE

File:

backend/app/models/evidence.py

Suggested fields:

id
anomaly_id
detector_name
score
reason
metadata
created_at

Possible detector names:

rule_qc
temporal
multivariate
isolation_forest
spatial
gru

The purpose is explainability.

---

# 20. SENSOR HEALTH TABLE

File:

backend/app/models/sensor_health.py

Suggested fields:

id
station_id
health_score
health_state
maintenance_status
last_updated
reason

Example health states:

healthy
warning
degraded
critical

Sensor health must be derived from ML/system evidence.

It must not be manually fabricated by frontend code.

---

# 21. PYDANTIC SCHEMAS

Create:

backend/app/schemas/

Recommended:

station.py
observation.py
anomaly.py
replay.py
explanation.py

Schemas define the API contract.

Example observation response concept:

{
  "station_id": "...",
  "timestamp": "...",
  "temperature_c": 25.1,
  "pressure_hpa": 1008.2,
  "relative_humidity_pct": 67.2
}

Example ML result:

{
  "station_id": "...",
  "timestamp": "...",
  "anomaly": true,
  "anomaly_score": 0.82,
  "confidence": 0.91,
  "classification": "...",
  "root_cause": "...",
  "detector_scores": {},
  "evidence": [],
  "sensor_health": {}
}

Use the actual existing ML field names when available.

Do not unnecessarily rename the ML output contract.

---

# 22. MOST IMPORTANT BACKEND FILE — ml_engine.py

File:

backend/app/services/ml_engine.py

This is the boundary between backend and ML.

Responsibilities:

1. Load ML artifacts.
2. Load models once.
3. Accept validated observation/context.
4. Call the existing ML pipeline.
5. Return structured ML results.

It must NOT:

- implement GRU
- implement Isolation Forest
- implement spatial reasoning
- implement temporal rules
- implement evidence fusion mathematics
- implement duplicate detector logic

The ML package remains responsible for those operations.

The backend adapter only connects to it.

---

# 23. ML ENGINE STARTUP BEHAVIOR

Models should not be loaded for every API request.

Bad:

Request
→ load model
→ predict
→ unload model

Correct:

Backend startup
→ load artifacts/models
→ keep them in memory
→ requests use loaded pipeline

This reduces latency.

If an artifact is missing:

Fail clearly during startup or return a clear service-unavailable state.

Do not silently return fake ML results.

---

# 24. DIRECT ML/BACKEND CROSS-CHECK

This test is mandatory.

Choose one fixed:

station_id
timestamp
temperature
pressure
humidity

Run it through the existing ML pipeline directly.

Save:

expected result

Then send the same observation through:

POST /api/observations/score

Save:

backend result

Compare:

anomaly
anomaly_score
confidence
classification
root_cause
detector scores

Expected:

Direct ML result ≈ Backend ML result

If they differ, stop integration and investigate.

Do not continue frontend development until this is understood.

---

# 25. OBSERVATION SERVICE

File:

backend/app/services/observation_service.py

Responsibilities:

- Validate observations.
- Store observations.
- Preserve provenance.
- Avoid accidental duplicates.
- Pass observations to ML when required.

Do not make this service another ML engine.

---

# 26. ANOMALY SERVICE

File:

backend/app/services/anomaly_service.py

Responsibilities:

- Receive ML result.
- Persist anomaly.
- Persist detector evidence.
- Persist root cause.
- Connect result to station/timestamp.
- Support retrieval.

Recommended logical flow:

observation
    ↓
ml_engine
    ↓
MLResult
    ↓
anomaly_service
    ↓
anomalies table
    +
evidence table
    +
sensor_health table

---

# 27. REPLAY SERVICE

File:

backend/app/services/replay_service.py

Purpose:

Turn historical data into a live-style demonstration.

Replay must:

- preserve chronological order
- use real observations
- support injected anomalies
- pass observations through the same ML path
- preserve station ID
- preserve timestamp
- avoid future leakage
- avoid duplicate inserts

The frontend should never generate fake anomaly values.

---

# 28. REPLAY API

Recommended:

POST /api/replay/start

POST /api/replay/stop

GET /api/replay/status

Optional:

POST /api/replay/reset

Possible replay configuration:

dataset
start_time
end_time
speed
station_ids

Replay state should be stored or controlled by backend.

---

# 29. API ROUTES

Recommended:

backend/app/api/routes/

health.py
stations.py
observations.py
anomalies.py
replay.py
explanations.py

Routes should remain thin.

Example:

HTTP request
    ↓
validate
    ↓
service
    ↓
database / ML
    ↓
response schema

Do not put long ML logic inside route files.

---

# 30. HEALTH API

Endpoint:

GET /api/health

Should confirm:

- backend is running
- database connection is available
- ML artifacts are available

Possible response:

{
  "backend": "ok",
  "database": "ok",
  "ml_engine": "ok"
}

Do not return "ok" if ML is actually unavailable.

---

# 31. STATION API

Recommended:

GET /api/stations

GET /api/stations/{station_id}

The frontend uses this for:

- station list
- station map
- station detail pages

---

# 32. OBSERVATION API

Recommended:

GET /api/observations

Parameters:

station_id
start
end

Optional:

POST /api/observations/score

This endpoint should pass data through the ML engine.

---

# 33. ANOMALY API

Recommended:

GET /api/anomalies

GET /api/anomalies/{anomaly_id}

Filters:

station_id
classification
start
end

The frontend should obtain anomaly information from this API.

---

# 34. SENSOR HEALTH API

Recommended:

GET /api/sensor-health/{station_id}

Return:

health score
health state
maintenance status
reason
last updated

The frontend must display the backend value.

---

# 35. GENAI INTEGRATION

GenAI is optional.

The core ML system does not require an LLM.

GenAI is used for:

- operator explanation
- natural-language reasoning
- maintenance assistance
- concise incident summaries

Correct architecture:

ML
 ↓
Structured evidence
 ↓
GenAI service
 ↓
LLM
 ↓
Explanation

Incorrect architecture:

Frontend
 ↓
LLM
 ↓
"Is this anomaly?"

Do not use the LLM as the primary detector.

---

# 36. GENAI SERVICE

File:

backend/app/services/genai_service.py

Input:

station
timestamp
T/P/RH
final anomaly decision
confidence
classification
root cause
detector scores
evidence
sensor health

Output:

Human-readable explanation.

Example:

"The observation was flagged primarily because the target station deviated from its recent temporal pattern while neighboring stations remained comparatively stable. The multivariate relationship between temperature and humidity also showed inconsistency. Sensor health is therefore classified as degraded."

The exact explanation must be generated from actual evidence.

Do not invent detector results.

---

# 37. GENAI SECURITY

Never put:

GENAI_API_KEY

inside React.

Never use:

VITE_GENAI_API_KEY

for a secret provider key.

The browser should call:

POST /api/explanations/anomaly/{id}

The backend then calls the LLM provider.

The key exists only on the backend.

Tests must mock the LLM provider.

If the LLM is unavailable:

Return a deterministic fallback explanation.

Anomaly detection must continue working.

---

# 38. GENAI AUTHORITY RULE

The LLM must NEVER modify:

anomaly
anomaly_score
confidence
classification
root_cause
sensor_health

ML is authoritative.

GenAI is explanatory.

---

# 39. FRONTEND STRUCTURE

Recommended:

frontend/

src/
├── api/
│   ├── client.js
│   ├── stations.js
│   ├── observations.js
│   ├── anomalies.js
│   ├── health.js
│   └── replay.js
│
├── components/
│   ├── StationMap.jsx
│   ├── ObservationChart.jsx
│   ├── AnomalyTimeline.jsx
│   ├── EvidencePanel.jsx
│   ├── SensorHealthCard.jsx
│   ├── ReplayControls.jsx
│   └── ExplanationPanel.jsx
│
├── pages/
│   ├── Dashboard.jsx
│   ├── StationDetails.jsx
│   └── AnomalyDetails.jsx
│
└── App.jsx

Use the project's existing React/Vite/Tailwind setup if already present.

---

# 40. FRONTEND RESPONSIBILITIES

Frontend responsibilities:

- display station map
- display weather observations
- display anomaly timeline
- display confidence
- display detector evidence
- display neighboring stations
- display sensor health
- control replay
- request GenAI explanation

Frontend must NOT:

- calculate anomaly score
- calculate confidence
- calculate root cause
- calculate sensor health
- call the LLM provider directly
- generate fake ML output

---

# 41. DASHBOARD DESIGN

The main dashboard should show:

1. Station map
2. Current station status
3. T/P/RH charts
4. Anomaly status
5. Confidence
6. Detector evidence
7. Neighbor comparison
8. Root cause
9. Sensor health
10. Maintenance recommendation
11. Replay controls
12. GenAI explanation

The dashboard should visually communicate:

"What happened?"

"Why was it detected?"

"Is it a weather event or sensor fault?"

"How confident is the system?"

"What should the operator inspect?"

---

# 42. STATION MAP

Use station coordinates from backend.

Do not hardcode station coordinates in React.

API:

GET /api/stations

Map displays:

- station marker
- current status
- health
- anomaly indicator

Clicking a station opens station details.

---

# 43. OBSERVATION CHARTS

Display:

Temperature
Pressure
Relative Humidity

Use backend data.

Recommended chart behavior:

- historical trend
- current observation
- anomaly marker
- time axis
- station selection

Do not alter raw values visually in a way that hides anomalies.

If corrected/imputed values are shown, clearly distinguish:

Raw
Predicted/Corrected

---

# 44. EVIDENCE PANEL

This is one of the most important UI components.

Show:

Rule QC
Temporal
Multivariate
Isolation Forest
Spatial
GRU

For each:

- score
- contribution
- explanation

Example:

Temporal:
High deviation from recent station behavior.

Spatial:
Target station differs from neighboring stations.

GRU:
High next-observation prediction error.

The actual text must come from backend evidence.

---

# 45. WEATHER EVENT VS SENSOR FAULT

The dashboard should clearly communicate the distinction.

Possible evidence:

Sensor fault:

Target station abnormal
+
Neighboring stations normal
+
Temporal inconsistency
+
Sensor health degradation

Weather event:

Multiple neighboring stations change similarly
+
Temporal behavior is coherent
+
Spatial consistency is high

This distinction must be based on existing ML evidence.

Do not create a frontend rule to determine it.

---

# 46. SENSOR HEALTH

Show:

Health score

Health state

Maintenance status

Reason

Example:

Health:
68/100

State:
Degraded

Maintenance:
Inspection recommended

Reason:
Repeated station-specific temporal deviations with weak neighboring-station support.

Again, backend/ML decides.

---

# 47. BACKEND TEST STRUCTURE

Recommended:

backend/tests/

conftest.py
test_ml_pipeline.py
test_spatial_features.py
test_spatial_reasoning.py
test_temporal_and_rules.py
test_database.py
test_api_health.py
test_api_stations.py
test_api_observations.py
test_api_anomalies.py
test_replay.py
test_genai_service.py

---

# 48. TESTING RULE

Every backend feature should have tests.

At minimum:

Database:

- connection
- insert
- retrieve

ML:

- artifact loading
- prediction
- output structure

API:

- status code
- validation
- response schema

Replay:

- chronological order
- station isolation
- duplicate protection

GenAI:

- provider success
- provider failure
- fallback
- cannot override ML

---

# 49. FRONTEND TESTING

At minimum:

npm install

npm run build

The production build must succeed.

Check:

- API URL
- environment variables
- no missing imports
- no runtime compile errors

Then run:

npm run dev

Test manually:

Dashboard
→ station
→ observation
→ anomaly
→ evidence
→ health
→ replay
→ explanation

---

# 50. COMPLETE ML + BACKEND TEST

This must be performed before frontend integration is considered complete.

Step 1:

Choose a fixed observation.

Step 2:

Run direct ML.

Step 3:

Call backend score endpoint.

Step 4:

Compare output.

Step 5:

Store result in PostgreSQL.

Step 6:

Retrieve result using anomaly API.

Step 7:

Confirm values match.

Expected:

Direct ML
=
Backend ML
=
Database result
=
API result

Only formatting may differ.

---

# 51. COMPLETE BACKEND + FRONTEND TEST

Start backend:

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

Start frontend:

cd frontend

npm run dev

Open dashboard.

Verify:

Frontend
→ API
→ Backend
→ PostgreSQL

Select station.

Verify real station data appears.

Do not use hardcoded demo data.

---

# 52. COMPLETE ML + BACKEND + FRONTEND TEST

This is the most important test.

Flow:

Replay
 ↓
Backend
 ↓
ML Engine
 ↓
ML Decision
 ↓
PostgreSQL
 ↓
FastAPI
 ↓
Frontend

Start replay.

Watch observations arrive.

Verify:

- timestamp advances
- station ID correct
- T/P/RH correct
- anomaly score changes appropriately
- evidence appears
- health updates
- database records appear
- frontend updates

Then inject/replay a known anomaly.

Verify it appears through the complete stack.

---

# 53. GENAI END-TO-END TEST

Trigger a known anomaly.

Backend stores:

ML decision
+
evidence

Frontend requests:

POST /api/explanations/anomaly/{id}

Backend:

loads anomaly
+
loads evidence
+
calls GenAI

Frontend receives explanation.

Verify:

The explanation references actual evidence.

Then disable/break GenAI configuration.

The anomaly system must still work.

A fallback explanation must be returned.

---

# 54. DOCKER IMPLEMENTATION

Final stack should contain:

PostgreSQL
Backend
Frontend

ML should be available inside the backend/application environment.

Do not create a separate ML microservice unless necessary.

Build:

docker compose build

Run:

docker compose up

Check:

docker compose ps

Logs:

docker compose logs backend

docker compose logs frontend

docker compose logs postgres

---

# 55. DOCKER DATABASE RULE

Inside Docker:

Do NOT normally use:

localhost

for PostgreSQL from the backend.

Use the Compose service name.

Example:

DATABASE_URL=postgresql+psycopg2://aws_app:<password>@postgres:5432/aws_anomaly_intelligence

Browser/frontend configuration is different from container-to-container networking.

---

# 56. FINAL END-TO-END ACCEPTANCE TEST

The system is not considered finished until this sequence works:

1. PostgreSQL starts.

2. Backend starts.

3. Backend reports database OK.

4. Backend reports ML engine OK.

5. Frontend starts.

6. Dashboard loads stations.

7. Station map displays real station data.

8. Observation data loads.

9. ML scoring works.

10. ML result is persisted.

11. Anomaly API retrieves the result.

12. Frontend displays the result.

13. Replay starts.

14. Replay remains chronological.

15. Known injected anomaly appears.

16. Detector evidence appears.

17. Neighbor comparison appears.

18. Sensor health updates.

19. GenAI explanation can be requested.

20. GenAI failure does not break ML.

21. Replay stops cleanly.

22. Docker reproduces the same behavior.

---

# 57. JUDGE DEMO FLOW

Recommended demonstration:

Dashboard
 ↓
Five stations
 ↓
Select target station
 ↓
Start historical replay
 ↓
Normal observations
 ↓
Controlled anomaly
 ↓
Detection
 ↓
Confidence
 ↓
Evidence
 ↓
Neighbor comparison
 ↓
Root cause
 ↓
Sensor health
 ↓
GenAI explanation

Then demonstrate a regional weather event:

Multiple stations change coherently.

System should recognize that the event is spatially consistent rather than blindly treating it as an isolated sensor failure.

This demonstrates the key intelligence of the system.

---

# 58. COMMON FAILURE MODES

Failure:

Backend cannot import torch.

Cause:

Wrong Python environment.

Fix:

Verify:

where python

python -c "import torch; print(torch.__version__)"

---

Failure:

Model file not found.

Cause:

Relative path depends on working directory.

Fix:

Use repository/project-root-aware paths.

---

Failure:

Backend returns fake anomaly values.

Cause:

Frontend/demo shortcut.

Fix:

Every displayed ML result must come from backend.

---

Failure:

LLM determines anomaly.

Cause:

Incorrect architecture.

Fix:

ML decision is authoritative.

---

Failure:

GenAI key appears in browser.

Cause:

Frontend provider integration.

Fix:

Move provider call into backend.

---

Failure:

Replay produces duplicate rows.

Cause:

No uniqueness/idempotency handling.

Fix:

Use station + timestamp + source/provenance appropriately.

---

Failure:

GRU sees future observations.

Cause:

Incorrect sequence construction.

Fix:

Keep inference causal.

---

Failure:

Sequence crosses stations.

Cause:

Global rolling window.

Fix:

Create sequences independently per station.

---

Failure:

Docker backend cannot connect to PostgreSQL.

Cause:

Using localhost.

Fix:

Use PostgreSQL Compose service name.

---

Failure:

Backend changed ML output.

Cause:

ML logic was duplicated/reimplemented.

Fix:

Use ml_engine adapter.

---

# 59. REQUIRED VALIDATION COMMANDS

ML:

python -m pytest -q

python -m compileall -q ml scripts backend

python scripts/prepare_data.py

python scripts/train_model.py

python scripts/inject_anomalies.py

python scripts/evaluate_model.py

Backend:

uvicorn backend.app.main:app --reload

Frontend:

npm install

npm run build

npm run dev

Docker:

docker compose build

docker compose up

docker compose ps

Git:

git status

git diff --check

---

# 60. GIT WORKFLOW

Before work:

git pull

git status

python -m pytest -q

After each logical component:

git diff --check

python -m compileall -q ml scripts backend

python -m pytest -q

git diff --stat

git status

Commit logical changes separately.

Examples:

Implement PostgreSQL persistence

Integrate ML engine with FastAPI

Implement replay API

Implement anomaly dashboard

Implement GenAI explanations

Do not make one giant commit containing unrelated changes.

---

# 61. WHAT MUST BE CROSS-CHECKED BEFORE MERGING

ML:

- tests pass
- artifacts load
- evaluation works

Database:

- connection works
- schema works
- CRUD works

Backend:

- imports work
- API starts
- ML loads
- database connects

Integration:

- direct ML = backend ML

Frontend:

- build succeeds
- API works
- real data displayed

GenAI:

- explanation works
- fallback works
- key protected
- cannot override ML

Docker:

- all services start
- services communicate
- complete replay works

---

# 62. DEFINITION OF DONE

The project is DONE only when:

ML
→ works independently

ML + Backend
→ produces the same decisions as direct ML

Backend + PostgreSQL
→ stores and retrieves decisions correctly

Backend + Frontend
→ displays real API data

ML + Backend + Frontend
→ supports real/injected replay

GenAI
→ explains evidence without changing ML decisions

Docker
→ reproduces the complete system

Testing
→ passes

Documentation
→ reflects actual implementation

---

# 63. FINAL ENGINEERING PRINCIPLE

The architecture must preserve a strict separation of responsibilities.

ML answers:

"What does the data indicate?"

Backend answers:

"How do we process, persist and expose that intelligence?"

PostgreSQL answers:

"What application state and results do we store?"

Frontend answers:

"How does the operator see and interact with the system?"

GenAI answers:

"How can we explain the existing evidence in natural language?"

No layer should silently take over another layer's responsibility.

The final product is:

Real observation
→ validated data
→ ML intelligence
→ explainable evidence
→ persisted decision
→ API
→ dashboard
→ optional GenAI explanation

That is the implementation path that must be followed from this point onward.