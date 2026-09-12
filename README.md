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

## Local Setup

```powershell
cd D:\aws-anomaly-intelligence
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

The local `.env` should use the Compose database:

```text
DATABASE_URL=postgresql+psycopg2://anomaly:anomaly@localhost:5432/anomaly_intelligence
```

Start the backend:

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Open the API docs at `http://localhost:8000/docs`.

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`.

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
GENAI_PROVIDER=openai
GENAI_MODEL=gpt-4o-mini
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

Run the full local product:

```powershell
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`

Compose passes the database hostname as `postgres` to the backend container. Do not use `localhost` for the database URL inside Docker.

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
