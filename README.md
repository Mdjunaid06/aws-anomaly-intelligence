# AWS Anomaly Intelligence

AWS Anomaly Intelligence is an explainable weather-station operations console. Real NOAA observations move through the frozen ML pipeline, PostgreSQL stores predictions and sensor health, FastAPI exposes the evidence, and the React dashboard supports investigation and replay.

## Architecture

```text
NOAA processed/features -> Replay -> PostgreSQL observation
                                      -> frozen ML pipeline
                                      -> prediction + evidence + health
                                      -> FastAPI -> React operations console
                                      -> optional grounded OpenAI explanation
```

The `ml/` directory is authoritative and must remain unchanged. GenAI explains stored ML results; it never decides anomalies or health.

## Prerequisites

- Python 3.12+
- Node.js 22+ and npm
- Docker Desktop with Compose

## Run ML, Backend, and Frontend (Windows PowerShell)

Run these commands from the repository root. The ML pipeline is not a separate server: prepare/train its artifacts once, then FastAPI loads and calls the existing ML engine for scoring and replay.

### 1. Create the Python environment and prepare ML artifacts

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/prepare_data.py
python scripts/train_model.py
```

The preparation step reads the raw NOAA files and creates processed observations and station metadata. Training creates the feature CSV and detector artifacts consumed by the backend. For the optional injected-anomaly evaluation, also run:

```powershell
python scripts/inject_anomalies.py
python scripts/evaluate_model.py
```

### 2. Start the database and backend

Configure `DATABASE_URL` in the root `.env`. For the Compose PostgreSQL service, use:

```text
DATABASE_URL=postgresql+psycopg2://anomaly:anomaly@localhost:5432/anomaly_intelligence
```

To use the committed local SQLite snapshot and its existing observations/predictions, set this instead:

```text
DATABASE_URL=sqlite:///./aws_anomaly.db
```

With SQLite, skip the PostgreSQL `docker compose` command. The snapshot is included in the repository. The generated processed data, feature CSV, and ML model artifacts are intentionally not tracked; recreate them with `prepare_data.py` and `train_model.py` above. Choosing PostgreSQL creates a separate, initially empty database; start replay to populate it.

Start PostgreSQL in one terminal:

```powershell
docker compose up -d postgres
```

Start FastAPI in another terminal from the repository root (activate `.venv` in that terminal first):

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`.

### 3. Start the frontend

In a third terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. The Vite app calls the backend at `http://localhost:8000` by default; override this with `VITE_API_BASE_URL` in `frontend/.env` if needed. Do not put backend secrets in the frontend environment.

### Verify

```powershell
python -m pytest -q
Push-Location frontend; npm run build; Pop-Location
```

The dashboard's system indicators show database, ML, replay, and optional GenAI status. Replay and observation scoring run through the backend and the same ML pipeline; there is no separate ML process to start.

### Expected UI on a fresh start

With the committed SQLite snapshot selected, processed data prepared, and models trained, the overview should show 6 station metadata entries, 1,004 stored observations, and 3 stored anomalies. The system header should report SQLite connected, ML online, replay stopped, and GenAI fallback unless a provider is configured. Metadata includes one station without observations, so replay configuration lists 5 stations while the map can show 6.

Replay state is held in backend memory and is not restored from the database. After starting or restarting the backend, open **Replay laboratory** and press **Start replay** (default speed: 5x) to see the timestamp, current weather values, classification, and progress advance. Stopping or restarting the backend resets replay progress but does not delete saved observations or predictions.

## Product Features

- Overview dashboard with real station health, anomaly records, and map data
- Anomaly investigation with complete ML evidence JSON
- Evidence-grounded fallback explanations when no LLM key is configured
- Optional OpenAI explanation and controlled operator assistant
- Replay Lab using `data/features/aws_features_2024_2025.csv`
- Replay controls: start, pause, resume, stop, reset
- PostgreSQL persistence through the same production batch/ML path

## Replay

Replay API:

```text
GET  /replay/config
GET  /replay/status
POST /replay/start
POST /replay/pause
POST /replay/resume
POST /replay/stop
POST /replay/reset
```

Example:

```powershell
Invoke-RestMethod http://localhost:8000/replay/start -Method Post -ContentType 'application/json' -Body '{"speed":5}'
Invoke-RestMethod http://localhost:8000/replay/status
```

Replay inserts real feature observations into PostgreSQL and invokes `BatchProcessingService`; it does not fabricate predictions or use a second detector.

## Explanation and Assistant APIs

```text
POST /assistant/explain
POST /assistant/chat
```

Example explanation request:

```json
{"prediction_id": 2}
```

With GenAI disabled, the backend returns a deterministic explanation derived from the stored anomaly, root cause, evidence, recommendation, and health. This keeps anomaly detection independent of OpenAI availability.

## OpenAI Setup

OpenAI is optional. Create a key from the official OpenAI developer platform at `https://platform.openai.com/api-keys` after signing in and selecting **Create new secret key**.

Put it only in the root backend `.env` file:

```text
GENAI_ENABLED=true
GENAI_PROVIDER=groq
GENAI_MODEL=openai/gpt-oss-20b
GENAI_API_KEY=your_key_here
```

Restart the backend after changing `.env`. Never place this key in `frontend/.env`, a `VITE_*` variable, source code, Docker image, or browser request. The frontend needs no provider key.

## API Surface

```text
GET  /stations/
GET  /observations/
GET  /observations/{id}
GET  /anomalies/
GET  /anomalies/{id}
GET  /health/
GET  /health/{station_id}
POST /batch/observation/{id}
```

## Docker

Generated processed data, features, and model artifacts are not tracked by Git. Create them from the raw NOAA files before building the backend image:

```powershell
python scripts/prepare_data.py
python scripts/train_model.py
```

Then run the full local product from the repository root:

```powershell
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`

Compose passes the database hostname as `postgres` to the backend container. Do not use `localhost` for the database URL inside Docker.

## Troubleshooting the Demo

- `GET /stations/` returns 503: station metadata is missing. Run `python scripts/prepare_data.py` from the repository root.
- `GET /replay/config` returns 400 with “Replay source not found”: the feature CSV is missing. Run `python scripts/train_model.py` after preparing data.
- `/system/status` reports the ML pipeline as error or no models loaded: train the models with `python scripts/train_model.py`, then restart FastAPI.
- The dashboard shows empty data or request errors: confirm FastAPI is listening on port 8000, check `VITE_API_BASE_URL`, and ensure the frontend origin is allowed by backend CORS.
- The header reports GenAI fallback: this is expected when GenAI is disabled or no backend key is configured; anomaly detection and deterministic explanations still work.
- Replay shows stopped at startup: this is expected because replay state is process-local. Start replay from Replay laboratory.
- Counts differ when using PostgreSQL: it is a separate, initially empty database. Start replay to populate it; use `sqlite:///./aws_anomaly.db` to inspect the committed local snapshot.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall backend ml scripts
git diff --check
git diff --name-only -- ml
Push-Location frontend; npm run build; Pop-Location
```

The final ML protection command must produce no output.

## SIH Demo Flow

1. Start PostgreSQL, backend, and frontend.
2. Open Replay Lab.
3. Start replay at a demo speed such as `5x`.
4. Watch real observations and predictions arrive in PostgreSQL.
5. Open a stored anomaly from the overview or investigation drawer.
6. Inspect confidence, classification, root cause, spatial evidence, and sensor health.
7. Select **Explain this anomaly** for the deterministic fallback or optional OpenAI explanation.

## Known Limitations

Replay state is held in the backend process and is intended for a single local demo process. Database schema creation currently uses SQLAlchemy `create_all`; a migration system can be added before production deployment.
