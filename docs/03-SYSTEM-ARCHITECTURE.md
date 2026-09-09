# AWS Anomaly Intelligence

## System Architecture

This document describes the practical system design for the SIH MVP. The architecture is intentionally simple, modular, and realistic for a student team to implement and demonstrate.

The final system is an AI-powered, explainable, spatiotemporal anomaly intelligence and predictive maintenance platform. Numerical models produce structured evidence; a grounded GenAI layer turns those results into human-readable explanations and investigation responses.

The finalized stack is Python 3.12/FastAPI/Uvicorn/Pydantic/SQLAlchemy with PostgreSQL for the backend, React.js/Vite/Tailwind CSS with Recharts and Leaflet for the frontend, and Docker Compose for local deployment. Data and ML work uses Pandas, NumPy, SciPy, scikit-learn, PyTorch, PyArrow, and SHAP.

---

## 1. Architecture Goal

The system is an AWS observation intelligence layer that continuously evaluates station readings and decides whether they look normal, anomalous, or likely linked to a real weather event.

The architecture supports:

- real AWS observations,
- historical replay for the SIH demo,
- real-time observation processing,
- controlled anomaly injection,
- rule-based and ML-based detection,
- spatial comparison across nearby stations,
- explainability,
- sensor health tracking,
- API-driven dashboard access,
- Docker-based local deployment.

The system should be modular, but it must stay simple enough to build and run within the project timeline.

---

## 2. High-Level Mental Model

```text
Historical/Real AWS Data
        ↓
Data Ingestion
        ↓
Validation + Preprocessing
        ↓
Data Quality / Rule-Based QC
        ↓
Feature Engineering
        ↓
Multiple Evidence Sources
        |-- Isolation Forest
        |-- GRU/LSTM temporal detection
        |-- PCA/Mahalanobis multivariate detection
        |-- Spatial/neighbor comparison
        ↓
Evidence Fusion and Anomaly Confidence
        ↓
Root-Cause Diagnosis
        ↓
Sensor Health and Maintenance Recommendation
        ↓
Structured Results
        ↓
Grounded GenAI Explanation Layer
        ↓
API
        ↓
Dashboard
```

This is the implementation-oriented baseline for the project.

The design should not require a separate model for every station. Instead, one shared detection pipeline operates on all stations with station-level context and neighbor comparison during inference.

---

## 3. Core Architectural Components

### 3.1 Historical/Real AWS Data

This layer represents the raw source data used by the system.

It includes:

- historical station observations,
- current sensor feeds,
- metadata such as station coordinates and elevation,
- optional replay datasets for demo and evaluation.

The key rule is that raw observations are never overwritten. Derived values are stored separately.

### 3.2 Data Ingestion

The ingestion layer receives observations from source files, APIs, or a demo replay engine.

It is responsible for:

- reading raw records,
- standardizing timestamps,
- assigning station IDs,
- preparing records for validation,
- routing data into the processing pipeline.

### 3.3 Validation + Preprocessing

Before any anomaly logic runs, records are validated.

Typical checks include:

- missing or null values,
- invalid timestamps,
- impossible numeric values,
- unit mismatches,
- duplicate records,
- unexpected station metadata,
- unrealistic step changes.

Preprocessing includes sorting records by time, normalizing formats, creating time windows, and preserving provenance. The original raw record remains untouched.

### 3.4 Feature Engineering and Anomaly Detection

This layer looks for suspicious patterns in each station’s observation history.

It is not a single black-box model. It combines:

- rule-based quality checks,
- Isolation Forest on engineered features,
- GRU/LSTM temporal detection experiments,
- PCA/Mahalanobis multivariate consistency detection,
- and spatial/neighbor comparison when coverage is valid.

The entire anomaly detection flow should work on a shared model or shared pipeline rather than a separate model per station.

### 3.5 Neighbor/Spatial Check

This component compares an active station with nearby stations when location information is available.

It produces a continuous spatial evidence object rather than a hard station-count decision. The object records total, usable, and missing neighbors; weighted agreement; distance and neighborhood weights; temporal alignment; direction and magnitude similarity; station reliability; data quality; elevation/context; geographic coherence; and common-mode risk.

Examples:

- if one station spikes while neighbors remain stable, the system gains evidence for a likely sensor issue,
- if all local stations move together, the system gains evidence for a credible regional event.

The spatial score is coverage-aware. A case with three usable stations out of five expected stations is not treated as 3/5 disagreement: missing observations are tracked separately and reduce confidence. A low-count but geographically coherent subset can support a localized event, while distant agreement remains weak. Station reliability reduces the influence of historically unreliable stations.

Common-mode checks inspect repeated exact values, identical sequences, suspiciously perfect correlation, stale-value propagation, ingestion duplication, and simultaneous source/message failure. Repeated sequence evidence can override an otherwise strong spatial consensus as a likely common-mode data fault.

This stage is important because it reduces false alarms and helps the system explain “why” an alert was raised.

### 3.6 Evidence Fusion

After temporal, multivariate, and spatial checks are computed, the system combines them into a single decision.

This stage should produce:

- anomaly score,
- severity estimate,
- confidence score,
- and a short evidence summary.

The fusion logic should be simple and transparent. It should not be a mysterious “black box” with no interpretable result.

Decision states are explicit: `normal`, `anomaly`, `likely_regional_event`, `likely_local_event`, `likely_sensor_fault`, `likely_sensor_drift`, `likely_stuck_sensor`, `likely_communication_fault`, `likely_common_mode_data_fault`, `inconclusive`, and `insufficient_spatial_evidence`. Confidence thresholds are centralized configuration values and must be tuned with validation data rather than scattered magic numbers.

### 3.7 Root Cause / Explanation

The anomaly detector flags suspicious behavior, but root cause classification is a separate layer.

This layer asks:

- Is the pattern more consistent with a sensor issue?
- Is it likely a communication/data problem?
- Is it more likely a genuine weather event?

The output must be framed carefully as a probable root cause, not an exact physical diagnosis.

### 3.8 Sensor Health

Sensor health is a longer-term view of station quality.

It tracks:

- repeated anomalies,
- anomaly severity,
- persistence over time,
- affected variables,
- recent station behavior.

This helps the system move from “there is a bad reading right now” to “this sensor is degrading or repeatedly misbehaving.”

### 3.9 Maintenance Recommendation

This layer converts sensor-health evidence into a practical recommendation such as monitor, review, calibrate, or inspect. It is a risk-based recommendation, not an automated physical diagnosis or repair command.

### 3.10 Structured Results and GenAI Explanation

The pipeline emits structured observations, evidence scores, anomaly confidence, probable root cause, sensor health, maintenance recommendation, and provenance for optional corrected values. GenAI receives only these structured results. It may explain anomalies, summarize stations or the network, and answer investigation questions, but it must not invent readings, evidence, or confidence values.

### 3.11 API

The API exposes observation, anomaly, and health data to the dashboard and other consumers.

This layer is intentionally small and practical for the MVP.

### 3.12 Dashboard

The dashboard is the human-facing layer for the project.

It shows:

- station overview,
- observation trends,
- anomalies,
- probable cause,
- sensor health,
- nearby station comparison,
- and replay/demo controls.
- grounded natural-language investigation.

---

## 4. Data Flow

The basic data flow is:

```text
Raw AWS observations
        ↓
Validation and schema checks
        ↓
Preprocessing and feature generation
        ↓
Data quality, feature engineering, and evidence stages
        ↓
Neighbor comparison and evidence fusion
        ↓
Decision, confidence, diagnosis, and structured results
        ↓
GenAI explanation / investigation
        ↓
Storage and API exposure
        ↓
Dashboard display
```

The system should always maintain a clear distinction between:

- RAW OBSERVATION
- PROCESSED OBSERVATION
- DETECTED ANOMALY
- INJECTED ANOMALY
- CORRECTED/IMPUTED VALUE

Raw values must not be overwritten.

Structured fusion results include anomaly status, classification, confidence, evidence scores, affected stations, supporting stations, contradicting stations, root cause, recommended action, and explanation facts. GenAI receives this object only after numerical processing.

---

## 5. Model Flow

The anomaly pipeline should be modular:

```text
Station history + observation window
        ↓
Feature engineering
        ↓
Rule-based QC
        ↓
Rule-based QC
        ↓
Isolation Forest score
        ↓
GRU/LSTM temporal score
        ↓
PCA/Mahalanobis multivariate score
        ↓
Neighbor score + spatial context
        ↓
Evidence fusion and anomaly confidence
        ↓
Root cause, health, maintenance, and structured result
```

This keeps the model understandable and allows experimentation without redesigning the whole app.

---

## 6. Prediction Flow

For each observation, the system should follow this decision flow:

1. Validate raw record.
2. Build the feature set from local station context.
3. Evaluate rule-based plausibility.
4. Run the evidence-source models.
5. Evaluate multivariate consistency.
6. Compare against nearby station behavior when coverage is sufficient.
7. Fuse evidence into anomaly confidence.
8. Produce diagnosis, sensor health, maintenance recommendation, and structured results.
9. Generate a grounded explanation when requested.

The final result should answer: “Why did the system consider this abnormal?”

---

## 7. Dashboard Flow

The dashboard should read from the API, not directly from raw storage.

```text
Dashboard
    ↓
API
    ↓
Detection and health services
    ↓
Database / storage layer
```

This keeps the presentation layer simple and decoupled from the detection logic.

---

## 8. Anomaly Injection and Replay

The architecture must support two modes.

### 8.1 Historical Replay Mode

This mode loads historical data and presents it as a time-ordered stream for demo purposes.

Workflow:

```text
Historical dataset
        ↓
Replay engine
        ↓
Stream observations over time
        ↓
Detection pipeline
        ↓
Dashboard
```

This mode is essential for the SIH demo because the project cannot depend on physical AWS hardware during presentation.

### 8.2 Real-Time Observation Mode

This mode is for live or near-live monitoring.

Workflow:

```text
Current station observation
        ↓
Validation
        ↓
Processing
        ↓
Anomaly detection
        ↓
Alert and health update
```

The same components should be reusable in both modes.

---

## 9. Storage at a High Level

The system needs a lightweight storage layer to support:

- raw observations,
- processed feature sets,
- detections,
- sensor health state,
- replay and evaluation data,
- and dashboard queries.

For the MVP, PostgreSQL is the relational storage layer. Files remain useful for raw, processed, feature, injected, and ground-truth datasets; PostgreSQL stores application observations, detections, health state, recommendations, and queryable structured results.

The important requirement is clear separation between raw data and derived outputs.

---

## 10. Docker at a High Level

Docker is used to package and run the application locally in a consistent environment.

For the MVP, Docker should handle:

- backend service,
- dashboard/web app,
- local database or data service,
- optional replay tooling,
- startup orchestration for demo scenarios.

This is enough to make the system easy to run on a laptop or judge environment without special cloud infrastructure.

The service composition remains intentionally small: one Python/FastAPI backend, one React/Vite frontend when implemented, and PostgreSQL managed with Docker Compose. No Kafka, Kubernetes, Spark, Redis, or microservice layer is required.

---

## 11. Implementation Realism

This architecture is realistic for a student team because it avoids unnecessary complexity.

It keeps the components small and understandable:

- ingestion,
- validation,
- feature/context generation,
- anomaly detection,
- spatial comparison,
- evidence fusion,
- explanation,
- sensor health,
- API,
- dashboard.

There is no requirement for microservices, Kafka, or Kubernetes for the MVP.

---

## 12. Final Design Principle

The strongest system is not the one with the most complex AI stack.

The strongest system is the one that:

- uses real data,
- validates observations carefully,
- compares against local and neighboring context,
- explains its decisions,
- and remains easy to demonstrate.

That is the design direction for AWS Anomaly Intelligence.

Imputation
Manual correction

This is important for reproducibility and scientific traceability.

9. Feature Engineering Layer

The detection system should not rely only on raw values.

Useful derived features can include:

Temporal features
current value
previous value
difference
percentage change
rolling mean
rolling standard deviation
rolling minimum
rolling maximum
rate of change

For example:

temperature_change =
current_temperature - previous_temperature
Cross-variable features

Relationships between:

Temperature
Pressure
Relative Humidity

can provide additional evidence.

For example, an unusual combination of variables may be more suspicious than an unusual value in isolation.

Time features

Potential features:

hour
day
month
season
day/night indicator

These can help the system learn normal temporal behavior.

10. Rule-Based Quality Control

Rule-based QC provides the first line of defense.

It is also an important baseline for evaluating the ML system.

Possible checks include:

10.1 Physical range check

Detect values outside physically meaningful limits.

Example:

Relative humidity < 0%
Relative humidity > 100%
10.2 Rate-of-change check

Detect unrealistic sudden changes.

Example:

Temperature:

20.1°C
20.3°C
20.2°C
48.7°C

The sudden change may indicate a sensor or data problem.

However, the system should not automatically classify every rapid change as a fault.

This is why additional evidence is required.

10.3 Persistence check

Detect values that remain unchanged for an unusually long period.

Example:

25.3
25.3
25.3
25.3
25.3
25.3
...

This may indicate a frozen sensor.

10.4 Missing-data check

Detect communication gaps.

Example:

10:00 → observation
10:01 → observation
10:02 → missing
10:03 → missing
10:04 → observation
10.5 Internal consistency check

Analyze whether the three variables show suspicious combinations.

11. Temporal Detection Module

The temporal detector analyzes how observations evolve over time.

Its objective is to answer:

"Does the current observation fit the station's recent temporal behavior?"

Potential approaches include:

LSTM Autoencoder
GRU Autoencoder
Forecasting model
Temporal reconstruction model
Statistical time-series methods

The model should operate on sequences rather than isolated observations.

Example:

t-4
t-3
t-2
t-1
t

The model learns normal temporal behavior.

An unusually high reconstruction or prediction error can become anomaly evidence.

12. Multivariate Detection Module

The multivariate detector analyzes the relationship between:

Temperature
Pressure
Relative Humidity

The objective is:

"Does this combination of variables look normal?"

Possible techniques include:

PCA
Mahalanobis distance
Multivariate statistical models
Autoencoders
Other suitable multivariate anomaly detectors

Example:

Temperature = normal
Pressure = normal
Humidity = highly inconsistent

A multivariate model may identify this relationship even if each individual value is technically inside its valid range.

13. Spatial Detection Module

The spatial module compares observations between nearby or relevant stations.

The main question is:

"Is this change happening only at this station, or is it occurring across multiple stations?"

Example:

Station A → Temperature suddenly increases by 12°C

Nearby Station B → normal
Nearby Station C → normal
Nearby Station D → normal

This increases suspicion that Station A may have a sensor problem.

But:

Station A → +12°C
Station B → +10°C
Station C → +11°C
Station D → +9°C

This may indicate a genuine regional meteorological event.

Therefore, spatial information helps reduce false alarms.

14. Neighbor Selection

The spatial module should not blindly compare every station with every other station.

Potential neighbor-selection criteria:

Geographical distance
Elevation similarity
Historical correlation
Station availability
Data quality

A station should only be used as a reliable reference when sufficient trustworthy data exists.

15. Neighbor Reliability

Not every neighboring station should have equal influence.

Each neighbor can receive a reliability score based on factors such as:

Recent data quality
Historical consistency
Missing-data rate
Sensor health
Distance
Correlation

Conceptually:

Neighbor A → reliability 0.95
Neighbor B → reliability 0.81
Neighbor C → reliability 0.42

Neighbor C should therefore contribute less to the spatial decision.

16. Evidence Fusion

This is one of the most important parts of the architecture.

Instead of:

IF ML model says anomaly
THEN anomaly

the system combines multiple evidence sources.

Possible evidence:

Rule QC score
Temporal anomaly score
Multivariate anomaly score
Spatial anomaly score
Persistence evidence
Neighbor agreement
Sensor-health history

Conceptual representation:

                RULE SCORE
                    |
                    |
TEMPORAL SCORE -----+ 
                    |
MULTIVARIATE SCORE -+----> EVIDENCE FUSION
                    |
SPATIAL SCORE ------+
                    |
HEALTH HISTORY -----+
                    |
                    v
             FINAL CONFIDENCE

The exact mathematical fusion method should be determined experimentally.

Possible approaches include:

Weighted scoring
Logistic regression
Calibrated probability model
Bayesian-style evidence combination
Learned fusion model

The selected method must be validated experimentally.

17. Anomaly Decision

The final anomaly engine should produce more than a Boolean value.

A conceptual output should contain:

station_id
timestamp

anomaly_detected
confidence_score

severity

evidence:
    rule_score
    temporal_score
    multivariate_score
    spatial_score

probable_root_cause

sensor_health

recommended_action

Example:

Anomaly detected: YES

Confidence: 0.94

Severity: HIGH

Likely cause:
Sensor malfunction

Evidence:
- sudden temperature jump
- neighboring stations remained stable
- temporal reconstruction error high
- persistence pattern detected

Sensor health:
42/100

Recommended action:
Inspect temperature sensor
18. Genuine Event vs Sensor Fault

This distinction is critical.

The system should attempt to differentiate:

Genuine Meteorological Event

from:

Station/Sensor/Data Anomaly

A simplified reasoning process:

Unexpected observation
        |
        v
Temporal evidence?
        |
        v
Multivariate evidence?
        |
        v
Spatial evidence?
        |
        v
Historical sensor behavior?
        |
        v
Evidence fusion
        |
        v
Likely genuine event / likely sensor issue / uncertain

The system should allow an uncertain category rather than forcing every event into a binary classification.

19. Root-Cause Diagnosis

After detecting an anomaly, the system should attempt to determine the probable reason.

Possible categories:

SPIKE
FLATLINE
DRIFT
MISSING / COMMUNICATION GAP
MULTIVARIATE INCONSISTENCY
SENSOR DEGRADATION
POSSIBLE REGIONAL EVENT
UNKNOWN

The diagnosis layer should use anomaly characteristics rather than only the anomaly score.

For example:

Long unchanged sequence
        ↓
Possible sensor freeze

or:

Gradual deviation from normal baseline
        ↓
Possible sensor drift

or:

All nearby stations change similarly
        ↓
Possible genuine regional event

The output should be described as a probable cause, not an absolute physical diagnosis.

20. Explainability Layer

The dashboard should explain why an observation was flagged.

The explanation should answer:

What happened?
Why was it suspicious?
Which evidence contributed?
How confident is the system?
What is the probable cause?
What should the operator do?

Example:

WHY FLAGGED?

1. Temperature changed 11.8°C within one interval.
2. Temporal model reconstruction error exceeded threshold.
3. Nearby stations did not show the same change.
4. Previous observations from this station were stable.
5. Sensor-health score has been declining.

Conclusion:
Likely temperature sensor anomaly.
Confidence: 93%.

Explainability should be understandable to a human operator.

21. Sensor Health Engine

The sensor-health engine maintains a continuously updated health state.

Conceptually:

100 ───────── Healthy
 80
 60 ───────── Warning
 40
 20 ───────── Critical
  0

Possible inputs:

Anomaly frequency
Anomaly severity
Persistence
Missing-data rate
Drift indicators
Root-cause history
Recent QC failures

The health score should not immediately collapse because of one isolated anomaly.

Repeated problems should have greater impact.

22. Predictive Maintenance

The sensor-health engine can be extended to identify sensors that may require maintenance.

Example:

Week 1 → Health 96
Week 2 → Health 92
Week 3 → Health 87
Week 4 → Health 79
Week 5 → Health 68

The system can identify the downward trend and generate:

Maintenance risk: HIGH

The objective is not to claim an exact failure date.

Instead, the system should provide an early warning that sensor behavior is deteriorating.

23. Optional Imputation Layer

The system may provide an estimated corrected value.

Example:

Observed temperature:
48.7°C

Estimated expected value:
24.9°C

Confidence:
0.91

However:

RAW VALUE ≠ CORRECTED VALUE

The raw observation must always remain available.

The corrected value should be stored separately with:

method
model_version
timestamp
confidence
reason

This ensures traceability.

24. Storage Architecture

The system should logically maintain separate categories of data.

RAW OBSERVATIONS
        |
        +---- PROCESSED OBSERVATIONS
        |
        +---- FEATURES
        |
        +---- DETECTION RESULTS
        |
        +---- EXPLANATIONS
        |
        +---- SENSOR HEALTH
        |
        +---- ANOMALY INJECTION RECORDS
        |
        +---- IMPUTATION RESULTS
        |
        +---- EXPERIMENT RESULTS

A suitable implementation may use:

PostgreSQL

for structured application data.

Large historical datasets can remain in:

CSV
Parquet
Object storage

depending on deployment requirements.

The exact database implementation can be finalized during development.

25. Backend API

The backend provides a bridge between the processing system and dashboard.

A conceptual API structure could include:

GET  /stations
GET  /stations/{station_id}
GET  /observations
GET  /anomalies
GET  /anomalies/{id}
GET  /sensor-health
GET  /explanations

POST /observations
POST /replay/start
POST /replay/stop
POST /anomaly-injection

These are conceptual endpoints.

The final API contract should be defined when the source-code architecture is finalized.

26. Real-Time Processing

The real-time pipeline should follow:

Incoming observation
        ↓
Validation
        ↓
Preprocessing
        ↓
Feature generation
        ↓
Rule checks
        ↓
Temporal model
        ↓
Multivariate model
        ↓
Spatial analysis
        ↓
Evidence fusion
        ↓
Diagnosis
        ↓
Sensor health update
        ↓
Store result
        ↓
Send result to dashboard

The system should process observations incrementally rather than retraining the complete model for every incoming observation.

27. Dashboard Architecture

The dashboard is the primary human-facing interface.

It should provide several views.

27.1 Network Overview

Show:

Total stations
Healthy stations
Warning stations
Critical stations
Active anomalies
Recent anomalies
27.2 Map View

Display stations geographically.

Conceptually:

       Station A ●
                  \
                   ● Station B
                      \
             ● Station C

Station markers can represent health/anomaly status.

27.3 Station Detail

Selecting a station should show:

Temperature
Pressure
Relative Humidity

Historical trends
Current anomaly status
Confidence
Sensor health
Recent alerts
27.4 Anomaly Detail

Display:

Timestamp
Station
Observed values
Anomaly confidence
Severity
Evidence
Root cause
Explanation
Recommended action
27.5 Sensor Health

Display health history over time.

Example:

Health Score

100 |████████████
 80 |██████████
 60 |███████
 40 |████
 20 |██
28. Replay Engine

The replay engine is important for the SIH demonstration.

Historical observations should be converted into a simulated stream.

Example:

Historical file
      ↓
Replay Engine
      ↓
Observation at 10:00
      ↓
Observation at 10:01
      ↓
Observation at 10:02
      ↓
...

The dashboard receives these observations as though they are arriving live.

This allows judges to see:

Observation arrives
        ↓
System processes it
        ↓
Anomaly detected
        ↓
Confidence calculated
        ↓
Explanation generated
        ↓
Dashboard updates
29. Anomaly Injection Engine

Because real datasets generally do not contain enough verified labels for every possible sensor failure, controlled anomaly injection should be used for evaluation.

The injection engine should create separate derived datasets.

Possible injected anomalies:

1. Spike
2. Flatline
3. Drift
4. Missing block
5. Communication gap
6. Multivariate corruption
7. Subtle anomaly

Example:

Original:

24.1
24.3
24.2
24.4
24.5

Injected:

24.1
24.3
48.9   ← injected spike
24.4
24.5

The system should maintain an injection record containing:

station_id
timestamp
anomaly_type
affected_variable
original_value
injected_value
injection_method

This provides ground truth for evaluation.

30. Genuine Event Test

The evaluation should not consist only of injected anomalies.

The system should also be tested on naturally occurring rapid changes.

The purpose is to measure false alarms.

Example:

Multiple nearby stations
        ↓
similar rapid temperature change
        ↓
system should recognize spatial consistency
        ↓
lower probability of isolated sensor fault

This is important because an anomaly detector should not treat every unusual meteorological event as sensor failure.

31. Experiment Tracking

Every experiment should record:

Dataset version
Feature version
Model version
Parameters
Thresholds
Training period
Testing period
Evaluation metrics
Results

This makes experiments reproducible.

Possible tools include:

MLflow

or another experiment-tracking system.

The exact tool can be finalized during implementation.

32. Model Versioning

Each trained model should have a version.

Example:

temporal_model_v1
temporal_model_v2
multivariate_model_v1
fusion_model_v1

The detection result should reference the model version that produced it.

Example:

model_version = temporal_model_v2

This becomes important when comparing historical results after a model update.

33. Error Handling

The system should fail safely.

Examples:

ML model unavailable

The system should still be able to run basic rule-based QC.

ML unavailable
      ↓
Rule QC continues
Neighbor station unavailable

Spatial evidence can be marked unavailable.

Spatial score = unavailable

It should not automatically classify the observation as normal.

Missing historical window

The temporal model should indicate insufficient context.

Invalid input

The observation should be rejected or quarantined.

34. Security and Configuration

Sensitive configuration should never be hardcoded.

Examples:

Database credentials
API keys
Secret keys
Deployment configuration

should be provided through environment variables.

A file such as:

.env.example

should document the required variables without containing real secrets.

35. Docker Architecture

The project should be containerized.

A conceptual deployment may contain:

                Docker Environment
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
     Backend        Frontend       Database
        |
        v
   ML Pipeline

Depending on implementation, the ML processing engine may run inside the backend service or as a separate worker.

Docker Compose can be used for local development.

Conceptually:

docker-compose
      |
      +---- backend
      |
      +---- frontend
      |
      +---- database
      |
      +---- optional worker

The final container structure should follow the actual implementation.

36. Scalability

The system should be designed so that adding more stations does not require creating a separate application for each station.

The preferred architecture is:

             Common Detection Pipeline
                       |
       +---------------+---------------+
       |               |               |
   Station A       Station B       Station C
       |               |               |
       +---------------+---------------+
                       |
                  Shared Models

Station-specific information can still be maintained through:

station metadata
historical baselines
sensor-health history
neighbor relationships
37. Multi-Station Processing

A scalable processing pattern is:

Incoming observations
        ↓
Group by station
        ↓
Create temporal context
        ↓
Run detection
        ↓
Perform spatial comparison
        ↓
Fuse evidence
        ↓
Generate result

The architecture should avoid loading the complete history of every station into memory for every prediction.

Only the required temporal window should normally be loaded into the online processing path.

38. Separation of ML and Application Logic

ML code and application code should remain separate.

For example:

ML layer
    ↓
returns scores/predictions
    ↓
Application layer
    ↓
interprets results
    ↓
API
    ↓
Dashboard

The dashboard should never directly contain ML logic.

Similarly, the ML model should not depend on frontend implementation.

This separation makes testing and future replacement easier.

39. Testing Strategy

The architecture should support testing at multiple levels.

Unit Tests

Test individual functions:

range check
missing-value check
feature calculation
anomaly score calculation
health score update
Integration Tests

Test:

Data ingestion
        ↓
Detection
        ↓
Database
        ↓
API
Model Tests

Evaluate:

Precision
Recall
F1
False Alarm Rate
Detection Latency
System Tests

Test complete scenarios:

Normal observation
        ↓
No false alarm

and:

Injected spike
        ↓
Anomaly detected
        ↓
Explanation generated
        ↓
Dashboard updated
40. Baseline Architecture

Before using complex ML models, the project should establish a baseline.

The baseline can be:

Rule-based QC

Then progressively add:

Rule QC
   ↓
Rule + Temporal
   ↓
Rule + Temporal + Multivariate
   ↓
Rule + Temporal + Multivariate + Spatial
   ↓
Full system + Health + Diagnosis

This allows the team to prove whether each component actually improves performance.

41. Ablation Architecture

Ablation testing should determine the contribution of each module.

Example:

Experiment A:
Rules only

Experiment B:
Rules + Temporal

Experiment C:
Rules + Temporal + Multivariate

Experiment D:
Rules + Temporal + Multivariate + Spatial

Experiment E:
Full architecture

Compare:

Precision
Recall
F1
False alarms
Detection latency

If the spatial component reduces false alarms, the team can demonstrate its practical value.

42. Recommended Build Order

The system should not be implemented all at once.

Recommended order:

Phase 1
Data ingestion + validation

        ↓

Phase 2
Preprocessing + feature engineering

        ↓

Phase 3
Rule-based QC

        ↓

Phase 4
Temporal detector

        ↓

Phase 5
Multivariate detector

        ↓

Phase 6
Spatial detector

        ↓

Phase 7
Evidence fusion

        ↓

Phase 8
Root-cause diagnosis

        ↓

Phase 9
Sensor health

        ↓

Phase 10
Replay + anomaly injection

        ↓

Phase 11
Backend API

        ↓

Phase 12
Dashboard

        ↓

Phase 13
Docker deployment

        ↓

Phase 14
Evaluation + optimization
43. MVP Architecture

The minimum working system should contain:

Real historical data
        ↓
Data processing
        ↓
Rule-based QC
        ↓
Temporal anomaly detection
        ↓
Multivariate detection
        ↓
Basic spatial comparison
        ↓
Evidence fusion
        ↓
Root-cause category
        ↓
Sensor health
        ↓
API
        ↓
Dashboard

The MVP should be strong enough to demonstrate the complete concept.

Advanced features can then be added.

44. Advanced Architecture

After the MVP is stable, advanced capabilities can include:

Better temporal models
Better spatial modeling
Probability calibration
Advanced explainability
Improved root-cause classification
Maintenance forecasting
Traceable imputation
Model monitoring
Automated retraining
Large-scale streaming

These should only be added after the core pipeline works reliably.

45. Important Architectural Principle

The system should never be presented as:

"We trained an AI model and it detects anomalies."

The stronger architecture is:

                OBSERVATION
                     |
                     v
             QUALITY CHECKS
                     |
                     v
          MULTIPLE AI/STATISTICAL
                DETECTORS
                     |
                     v
              SPATIAL CONTEXT
                     |
                     v
             EVIDENCE FUSION
                     |
                     v
             ANOMALY DECISION
                     |
          +----------+----------+
          |                     |
          v                     v
       DIAGNOSIS          SENSOR HEALTH
          |                     |
          +----------+----------+
                     |
                     v
               EXPLANATION
                     |
                     v
               OPERATOR UI

This makes the system an intelligent observation-quality and sensor-health platform, rather than merely an anomaly classifier.

46. Final Architecture Mental Model

The complete project can be remembered using:

COLLECT
   ↓
VALIDATE
   ↓
UNDERSTAND
   ↓
DETECT
   ↓
COMPARE
   ↓
FUSE
   ↓
EXPLAIN
   ↓
DIAGNOSE
   ↓
TRACK
   ↓
ACT

Where:

COLLECT  → obtain AWS observations

VALIDATE → verify data quality

UNDERSTAND → generate temporal and multivariate context

DETECT → identify abnormal observations

COMPARE → compare with temporal history and neighboring stations

FUSE → combine multiple evidence sources

EXPLAIN → show why the observation was flagged

DIAGNOSE → estimate the probable root cause

TRACK → maintain sensor-health history

ACT → provide maintenance/review recommendations
47. Final Architecture Definition

The AWS Anomaly Intelligence system is a modular, real-time anomaly detection and observation-quality architecture for Automatic Weather Station networks.

Its core architecture is:

Real/Historical AWS Data
          ↓
Data Ingestion
          ↓
Validation
          ↓
Preprocessing
          ↓
Feature Engineering
          ↓
Rule-Based QC
          ↓
Temporal Detection
          ↓
Multivariate Detection
          ↓
Spatial Detection
          ↓
Evidence Fusion
          ↓
Anomaly Decision
          ↓
Root-Cause Diagnosis
          ↓
Sensor Health
          ↓
Explainable Result
          ↓
Backend API
          ↓
Dashboard

The architecture intentionally combines conventional quality control, statistical methods, machine learning, temporal context, spatial context, explainability, and sensor-health tracking.

The system should be evaluated experimentally rather than assuming that every additional component improves performance.

48. Important Note About Source-Code Structure

The exact source-code folder architecture is intentionally not finalized in this document.

The final structure should be created after the development team decides:

Backend framework
Frontend framework
ML framework
Database
Streaming mechanism
Model-serving approach
Docker services
Testing structure
Deployment structure

Once the final source-code architecture is provided, it should be documented separately and used as the implementation contract for the team.

This prevents the documentation from inventing folders or modules that are never actually implemented.

End of Document

### After pasting
