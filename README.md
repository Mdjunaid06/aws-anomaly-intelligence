# AWS Anomaly Intelligence

An explainable anomaly-quality system for observations from multiple Automatic Weather Stations (AWS).

## Project Scope

The project will use real historical meteorological observations and preserve the data lifecycle:

```text
data/raw/ -> data/processed/ -> data/features/
						 -> data/injected/ + data/ground_truth/
```

The planned detection approach combines rule-based meteorological quality checks, Isolation Forest, multivariate consistency analysis, temporal models, spatial context, evidence fusion, root-cause classification, and sensor health scoring. Implementation and experiments are intentionally not included in this structure-initialization step.

The architecture supports multiple stations and observation fields including `station_id`, `timestamp`, coordinates, temperature, atmospheric pressure, and relative humidity.

## Repository Layout

- `data/`: raw, processed, feature, injected, and ground-truth data stages
- `ml/`: notebooks, reusable ML source areas, and model/metric artifacts
- `backend/`: reserved FastAPI application and tests
- `frontend/`: reserved React/Vite application; not initialized yet
- `scripts/`: reproducible data and ML pipeline entry points
- `docs/`: project, architecture, data, ML, API, and team documentation

## Development Status

This repository currently contains the project structure and essential configuration only. Application logic, model implementations, datasets, database migrations, and frontend files will be added in later phases.

## Data Rules

- Files in `data/raw/` are immutable source data.
- Anomaly injection operates on processed copies and writes to `data/injected/`.
- Injection metadata is stored separately in `data/ground_truth/`.
- Do not commit credentials, downloaded datasets, trained model artifacts, or experiment outputs.

## Configuration

Copy `.env.example` to `.env` and adjust local values when implementation begins. Python dependencies are listed in `requirements.txt`; frontend dependencies will belong to `frontend/package.json` when the React/Vite application is initialized.
