# AWS Anomaly Intelligence

## Application and API

This document defines the practical MVP application and the minimal API needed for the dashboard and demo. The design is intentionally small, clean, and directly aligned with the SIH problem.

The application stack is React.js/Vite/Tailwind CSS with Recharts, Leaflet, React-Leaflet, and Axios on the frontend, and Python 3.12/FastAPI/Uvicorn with Pydantic, SQLAlchemy, PostgreSQL, and python-dotenv on the backend. Docker Compose provides local orchestration.

---

## 1. Dashboard Purpose

The dashboard is the primary demonstration and inspection interface for the project.

Its purpose is to let a user:

- view the network of AWS stations,
- inspect station observations,
- see anomalies as they occur,
- review evidence for a flagged anomaly,
- understand probable root cause,
- monitor sensor health,
- review maintenance recommendations,
- and replay historical data in a demo mode.

The dashboard should be simple enough for an SIH judge to understand in under five minutes.

---

## 2. Main Dashboard Screen

The main screen should include:

- a network overview map or station grid,
- current station status summary,
- anomaly count by station,
- latest observations for Temperature, Pressure, and Humidity,
- recent anomaly list,
- quick filters by station and time range,
- and a button to start replay or demo mode.

This is the central dashboard view for the prototype.

---

## 3. Station Overview

The station overview shows a single station’s latest status.

It should include:

- station_id,
- station location,
- last updated timestamp,
- latest temperature,
- latest pressure,
- latest relative humidity,
- health status,
- anomaly flag,
- recent trend summary.

This lets a user know whether a station looks normal or suspicious at a glance.

---

## 4. Station Detail

The station detail page should provide a deeper view for a selected station.

It includes:

- historical trend chart for temperature,
- historical trend chart for pressure,
- historical trend chart for relative humidity,
- anomaly timeline,
- recent rule-based warnings,
- ML anomaly score trend,
- neighboring stations comparison,
- and sensor health summary.

This is where the user can inspect evidence behind the anomaly decision.

---

## 5. Anomaly Detail

The anomaly detail screen should explain the specific flagged result.

It should include:

- anomaly_id,
- station_id,
- timestamp,
- affected variable(s),
- severity,
- confidence,
- anomaly type,
- evidence summary,
- comparison to historical station behavior,
- comparison to neighboring stations,
- and probable root-cause classification.

This page directly supports the project’s explainability requirement.

The explanation is generated from structured pipeline evidence. A grounded GenAI layer may turn that result into human-readable language, but it must not invent readings, confidence, or evidence.

---

## 6. Sensor Health

The sensor health area should show whether a station is healthy, degraded, or critical.

The view may include:

- historical health score,
- recent anomaly count,
- recent persistence trend,
- quality category,
- maintenance recommendation such as monitor, review, calibrate, or inspect.

Maintenance recommendations are evidence-based and are not automated repair commands.

This is not a predicted physical diagnosis, but a practical health tracking feature.

---

## 7. Map Visualization

The dashboard should include a simple map or layout showing stations across the region.

Each station can be color-coded by status:

- normal,
- warning,
- critical anomaly,
- degraded sensor health,
- or regional event pattern.

This helps a user understand whether a warning is isolated or part of a broader pattern.

---

## 8. Historical Trend Charts

The dashboard should show time-series plots for:

- temperature,
- pressure,
- relative humidity,
- anomaly overlays,
- and neighboring-station comparisons.

The charts should be easy to read and focused on the SIH demo rather than advanced analytics.

## 9. Natural-Language Investigation

The dashboard may provide a natural-language investigation surface backed by the GenAI explanation layer. Supported tasks include:

- explain a selected anomaly,
- summarize a station’s recent behavior,
- summarize current network-wide anomaly status,
- explain why evidence supports a likely local fault or regional event.

GenAI receives structured observations, evidence scores, anomaly confidence, root cause, sensor health, maintenance recommendation, and provenance. It must state insufficient evidence when the structured results do not support an answer.

---

## 10. Replay/Demo Mode

Replay mode is essential for the SIH prototype.

The user should be able to:

- load historical data,
- start a replay,
- pause or stop replay,
- inject or simulate an anomaly,
- observe the system react in near real time,
- and view the resulting alert and explanation.

This allows the project to demonstrate the logic without requiring physical AWS sensors.

---

## 10. Small API Contract

The backend should expose a minimal API for the dashboard and demo control.

The following contract is appropriate for the MVP.

### GET /stations

Purpose: return all known AWS stations and their metadata.

Request:

- no body required

Response:

```json
{
  "stations": [
    {
      "station_id": "ST001",
      "name": "Station 1",
      "latitude": 12.97,
      "longitude": 77.59,
      "elevation": 920.0,
      "status": "normal"
    }
  ]
}
```

Important fields:

- station_id
- latitude
- longitude
- elevation
- status

### GET /stations/{station_id}

Purpose: return the current status and summary for one station.

Request:

- station_id in path

Response:

```json
{
  "station_id": "ST001",
  "latest_observation": {
    "timestamp": "2025-01-01T10:15:00Z",
    "temperature": 31.2,
    "pressure": 1013.4,
    "relative_humidity": 56.1
  },
  "health": "degraded",
  "anomaly_count": 3
}
```

Important fields:

- latest_observation
- health
- anomaly_count

### GET /observations

Purpose: return historical or current observations with filtering support.

Request:

- optional query parameters such as station_id, start_time, end_time, limit

Response:

```json
{
  "observations": [
    {
      "timestamp": "2025-01-01T10:00:00Z",
      "station_id": "ST001",
      "temperature": 30.1,
      "pressure": 1012.8,
      "relative_humidity": 55.7
    }
  ]
}
```

Important fields:

- timestamp
- station_id
- temperature
- pressure
- relative_humidity

### GET /anomalies

Purpose: return the list of detected anomalies.

Request:

- optional filters by station_id, start_time, end_time, severity

Response:

```json
{
  "anomalies": [
    {
      "anomaly_id": "A-1001",
      "station_id": "ST001",
      "timestamp": "2025-01-01T10:15:00Z",
      "severity": "high",
      "confidence": 0.92,
      "status": "detected"
    }
  ]
}
```

Important fields:

- anomaly_id
- station_id
- timestamp
- severity
- confidence
- status

### GET /anomalies/{id}

Purpose: return the detailed explanation for a single anomaly.

Request:

- anomaly_id in path

Response:

```json
{
  "anomaly_id": "A-1001",
  "station_id": "ST001",
  "timestamp": "2025-01-01T10:15:00Z",
  "affected_variable": "temperature",
  "severity": "high",
  "confidence": 0.92,
  "probable_root_cause": "likely sensor issue",
  "evidence": {
    "temporal": "strong deviation from recent history",
    "multivariate": "temperature change inconsistent with pressure/humidity trend",
    "spatial": "nearby stations stable"
  }
}
```

Important fields:

- probable_root_cause
- evidence
- severity
- confidence

### GET /sensor-health

Purpose: return the current health score or health state for all stations.

Request:

- optional station_id filter

Response:

```json
{
  "sensor_health": [
    {
      "station_id": "ST001",
      "health_score": 0.74,
      "status": "degraded",
      "anomaly_count_7d": 3
    }
  ]
}
```

Important fields:

- station_id
- health_score
- status
- anomaly_count_7d

### POST /observations

Purpose: submit a new observation for ingestion and processing.

Request:

```json
{
  "timestamp": "2025-01-01T10:15:00Z",
  "station_id": "ST001",
  "temperature": 31.2,
  "pressure": 1012.8,
  "relative_humidity": 55.7
}
```

Response:

```json
{
  "status": "accepted",
  "observation_id": "OBS-1234"
}
```

Important fields:

- timestamp
- station_id
- temperature
- pressure
- relative_humidity
- status

### POST /replay/start

Purpose: begin historical replay for the demo.

Request:

```json
{
  "dataset_id": "demo-jan-2025",
  "speed": "1x"
}
```

Response:

```json
{
  "status": "replay_started",
  "dataset_id": "demo-jan-2025"
}
```

### POST /replay/stop

Purpose: stop the demo replay.

Request:

- no body required

Response:

```json
{
  "status": "replay_stopped"
}
```

---

## 11. How the Dashboard, API, Detection Pipeline, and Database Work Together

The flow is:

```text
Dashboard
    ↓
API
    ↓
Detection Pipeline
    ↓
Database / storage layer
```

In practice:

1. The dashboard requests station and anomaly data from the API.
2. The API reads from the database and/or calls the detection layer for live results.
3. The detection pipeline validates input observations, computes features, checks anomaly logic, and updates anomaly and sensor-health records.
4. The database stores raw observations, processed data, detections, and health state.
5. The dashboard queries this combined state and presents a simple visual explanation.

This keeps the architecture small and provides a clean MVP implementation path.

---

## 12. MVP Implementation Guidance

The team should not over-engineer the API.

The API should be:

- small,
- REST-style,
- clear,
- and focused on dashboard, demo, health, maintenance, and grounded investigation use.

Natural-language questions may be handled through the application’s investigation flow using structured results from the existing API and backend services. GenAI must preserve station IDs, timestamps, confidence, evidence, root cause, health, and maintenance information, and must state insufficient evidence instead of inventing facts.

There is no need for a large microservice ecosystem for this project. A single backend application is enough for the MVP.
