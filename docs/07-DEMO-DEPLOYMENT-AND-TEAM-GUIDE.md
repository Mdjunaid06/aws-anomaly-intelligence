# AWS Anomaly Intelligence

## Demo Deployment and Team Guide

This document is intended to help a new teammate understand the project, run the MVP locally, and contribute without creating confusion or conflicting changes.

---

## PART A — PROJECT SETUP

### Prerequisites

The finalized implementation stack is:

- Git
- Python 3.12
- FastAPI, Uvicorn, Pydantic, SQLAlchemy, and python-dotenv
- PostgreSQL
- React.js, Vite, Tailwind CSS, Recharts, Leaflet, React-Leaflet, and Axios
- Pandas, NumPy, SciPy, scikit-learn, PyTorch, PyArrow, and SHAP
- pytest and HTTPX
- Docker and Docker Compose
- a local code editor or IDE
- data source access to real AWS/weather observations
- a browser for the dashboard

### Cloning the repository

To clone the project:

- clone the repository from the project source
- open the workspace in the IDE
- confirm that the repository includes the project documentation, source code, and Docker configuration

Exact repository URL: TBD — to be finalized during implementation.

### Environment setup

The project should use a lightweight local environment for development and demo work.

Setup steps should include:

- creating a local virtual environment or equivalent project environment,
- installing project dependencies,
- setting environment variables for database access, local app settings, and replay settings,
- and verifying that the app starts without runtime errors.

### Environment variables

Environment variables should include any required values such as:

- application port,
- database path or connection string,
- data directory paths,
- demo mode flags,
- optional debug settings.

Exact variable names: TBD — to be finalized during implementation.

### Installing dependencies

Use the project’s package manager or environment file once the implementation exists.

Installation step: TBD — to be finalized during implementation.

### Running locally

The application should run locally using the project’s standard startup commands.

Local startup command: TBD — to be finalized during implementation.

---

## PART B — DOCKER

### What Docker is being used for

Docker is used to make the app easy to run locally and in a demo environment.

It can package:

- the backend service,
- dashboard/web interface,
- local data or database layer,
- and the replay/demo tooling.

### How to build

Build command: TBD — to be finalized during implementation.

Typical placeholder:

- docker build -t aws-anomaly-intelligence .

The exact command must be confirmed when the implementation exists.

### How to start

Start command: TBD — to be finalized during implementation.

Typical placeholder:

- docker compose up --build

### How to stop

Stop command: TBD — to be finalized during implementation.

Typical placeholder:

- docker compose down

### How to check logs

Log command: TBD — to be finalized during implementation.

Typical placeholder:

- docker compose logs -f

---

## PART C — APPLICATION USAGE

### How to start the application

Start the backend and dashboard using the project’s local startup method or Docker process.

Startup steps: TBD — to be finalized during implementation.

### How to access the dashboard

The dashboard should be available through the configured local app port.

Dashboard URL: TBD — to be finalized during implementation.

### How to load data

Use the project’s demo or ingestion workflow to load real historical AWS data into the app.

Data-load step: TBD — to be finalized during implementation.

### How to run replay

Replay mode should be triggered from the dashboard or an API endpoint.

Replay trigger: TBD — to be finalized during implementation.

### How to view anomalies

Open the anomalies page or dashboard list to review:

- anomaly time,
- station ID,
- confidence,
- severity,
- evidence,
- and probable root cause.

### How to inspect sensor health

Use the sensor health section of the dashboard to review:

- health score,
- recent anomaly history,
- station degradation trend,
- and recent quality status.

---

## PART D — DATA TEAM GUIDE

The data team is responsible for:

- identifying and downloading real AWS observation datasets,
- validating columns, units, and metadata,
- checking duplicates and missing values,
- separating raw and processed data,
- documenting the data provenance,
- creating the anomaly injection dataset for evaluation,
- and ensuring the final dataset is suitable for training and demo work.

The data team must work carefully to avoid fabricating or mislabeling data.

---

## PART E — ML TEAM GUIDE

The ML team is responsible for:

- building the feature engineering pipeline,
- implementing the baseline and rule-based QC checks,
- selecting and validating the rule-based QC, Isolation Forest, GRU/LSTM, PCA/Mahalanobis, and spatial evidence sources,
- creating the spatial context logic,
- validating the evidence fusion rule,
- validating sensor-health and maintenance recommendations,
- measuring Precision, Recall, F1, False Alarm Rate, and Detection Latency,
- and documenting the experiments and ablation results.

The ML team should prefer practical, explainable solutions over overly complex models.

---

## PART F — BACKEND TEAM GUIDE

The backend team is responsible for:

- data ingestion and validation,
- storing observations and derived outputs,
- exposing the API,
- handling replay controls,
- connecting detection results to dashboard queries,
- and ensuring the system remains simple and reliable.

The backend should not become a distributed system for the MVP. The goal is a clear and maintainable implementation that works end-to-end.

The backend also owns structured anomaly results, evidence provenance, sensor health, maintenance recommendations, and the grounded GenAI context. GenAI must not replace numerical detection or invent unsupported facts.

---

## PART G — FRONTEND TEAM GUIDE

The frontend team is responsible for:

- designing the dashboard layout,
- showing station and anomaly information clearly,
- building the station detail and anomaly detail views,
- representing map status and historical trends,
- presenting maintenance recommendations and grounded natural-language investigation,
- and making the demo easy to present.

The UI should prioritize clarity and demo-friendly visuals over unnecessary complexity.

---

## PART H — DEMO FLOW

Below is the recommended 5-minute SIH demo flow.

### 1. Show network overview

Open the dashboard and show the AWS station network.

### 2. Select a station

Choose a single station that is currently behaving normally.

### 3. Show normal weather observations

Display normal Temperature, Pressure, and Relative Humidity values for that station.

### 4. Start historical replay

Begin the replay or dataset playback so the observations appear to arrive over time.

### 5. Inject or show a sensor anomaly

Trigger a controlled anomaly such as a spike or flatline at the selected station.

### 6. System detects it

The dashboard should show that an anomaly has been detected.

### 7. Show confidence and evidence

Display the model’s confidence and the main reasons behind the alert.

### 8. Show probable root cause

Explain the result as a probable root cause, such as a likely sensor issue or possible data problem.

### 9. Show sensor health degradation

Highlight the sensor health score or health status trend over time.

### 10. Show neighboring stations

Display surrounding stations to prove whether the pattern is isolated or part of a broader regional event.

### 11. Demonstrate the correct distinction

Show that the system can distinguish between:

- a likely isolated sensor issue, and
- a possible genuine regional weather event.

This demonstrates the project’s core value over a simple threshold-based system.

### 12. Demonstrate grounded explanation

Ask the investigation interface to explain the selected result or summarize the station. Confirm that the response cites only structured observations, evidence, confidence, root cause, health, and maintenance information, and states when evidence is insufficient.

---

## PART I — TEAM WORKFLOW

### Git branches

Use simple and clear branch names such as:

- data/
- ml/
- backend/
- frontend/
- demo/

Keep the branch names descriptive and short.

### Commits

Each commit should be focused and meaningful.

Good practice:

- one logical change per commit,
- descriptive commit messages,
- and commit only the files related to the current task.

### Pull requests

Open a pull request for each major feature or bugfix.

The PR should explain:

- what changed,
- why it changed,
- and how it relates to the project scope.

### Issue assignment

Use issues for tasks such as:

- dataset validation,
- ML experiment tracking,
- dashboard component work,
- API endpoints,
- Docker setup,
- and demo preparation.

### Documentation updates

Any significant change should also update the relevant documentation file.

### Avoiding conflicts

To avoid merge conflicts:

- communicate before editing shared files,
- keep branch scope narrow,
- and update documentation when project direction changes.

No complicated Git workflow is required for this project.

---

## PART J — FINAL DEMO CHECKLIST

Before the final SIH demo, confirm the following:

- data working
- model working
- API working
- dashboard working
- replay working
- anomaly injection working
- explanation working
- sensor health working
- Docker working
- README updated

This checklist should be reviewed before every demo rehearsal.

---

## Final Note

This project should be built as a practical, explainable, and demonstrable MVP. The goal is not to create a large enterprise architecture. The goal is to show that the team can reliably detect suspicious AWS observations, distinguish them from real weather effects, explain the decision, and present the system clearly to judges.
