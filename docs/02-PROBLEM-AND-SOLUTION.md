# AWS Anomaly Intelligence

## Problem and Solution

This document defines the SIH problem, the actual work we are solving, the MVP scope, and the practical reasoning behind the system.

---

## 1. SIH Problem Statement

**Problem ID:** SIH26073

**Problem title:** AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

**Organization:** Ministry of Earth Sciences (MoES)

**Theme:** Disaster Management

The system must detect abnormal, inconsistent, or faulty observations from multiple Automatic Weather Stations and decide whether an unusual reading is more likely to be:

- a genuine weather event,
- a sensor/data issue,
- a communication or transmission problem,
- or a degradation in station behavior.

The three core AWS variables we are concerned with are:

- Temperature
- Atmospheric Pressure
- Relative Humidity

---

## 2. Actual Problem We Are Solving

The core challenge is not simply “find a value that is different.”

The actual problem is:

> Given multiple AWS stations, can we determine whether an unusual observation is a real meteorological event or a likely sensor/data fault?

A simple example:

- ST001: 55.0°C
- ST002: 30.9°C
- ST003: 31.1°C
- ST004: 30.8°C

The 55.0°C reading is suspicious, but it may still be a real regional heat event if all nearby stations also rise together. Therefore, the system must combine multiple types of evidence before making a final decision.

---

## 3. Why Simple Threshold-Based Anomaly Detection Is Insufficient

A threshold-only rule such as “if temperature > 45°C, flag anomaly” is not enough.

Why:

- A real weather extreme can legitimately exceed a historical threshold.
- A sensor fault may produce a value that looks “normal” in some local range but is inconsistent with station behavior.
- A sudden unusual value may be due to a single bad reading, not a full sensor failure.
- Pressure and humidity can change simultaneously in ways that are meaningful only when considered together.

Therefore, a threshold is useful as one signal, but it cannot be the complete decision logic.

---

## 4. Proposed Solution

The project follows a layered reasoning pipeline:

DATA
→ VALIDATION
→ FEATURE/CONTEXT GENERATION
→ ANOMALY DETECTION
→ SPATIAL/NEIGHBOR CHECK
→ FINAL DECISION
→ EXPLANATION
→ SENSOR HEALTH
→ DASHBOARD

This is the practical MVP design.

The system uses real AWS observations, validates them, checks local temporal behavior, compares them with related stations, and then decides whether the most likely explanation is:

- normal weather,
- possible sensor issue,
- likely regional event,
- or data quality problem.

The system does not rely on a single model or a single threshold. It combines evidence from multiple sources before deciding.

---

## 5. What the System Receives

The system receives observations from multiple AWS stations. Each observation contains at least:

- timestamp
- station_id
- temperature
- atmospheric pressure
- relative humidity
- latitude
- longitude
- elevation

The system may also receive station metadata and historical observations for comparison.

Important: all raw observations are preserved as raw data. The system should never overwrite the original value when it creates a corrected or imputed record.

---

## 6. What the System Detects

The system detects suspicious observation patterns, such as:

- sudden spike
- flatline or frozen sensor behavior
- drift or gradual degradation
- missing block or communication loss
- multivariate inconsistency across temperature, pressure, and humidity
- abrupt deviation from a station’s own historical pattern
- abnormal comparison with nearby stations

These events are treated as DETECTED ANOMALIES, not final proof of sensor failure.

---

## 7. What the System Outputs

The system produces an explainable anomaly record with:

- station_id
- timestamp
- affected variable or variables
- anomaly type
- severity
- confidence
- evidence summary
- probable root cause
- sensor health impact

Example output:

- Station: ST001
- Timestamp: 2025-01-01 10:15
- Variable: Temperature
- Status: DETECTED ANOMALY
- Severity: Medium/High
- Confidence: 90%
- Likely explanation: possible temperature sensor issue
- Evidence: temporal deviation, multivariate inconsistency, neighboring stations normal

The output is not just “anomaly=yes.” It explains why the system flagged the observation.

---

## 8. How the System Distinguishes a Sensor Fault from a Regional Weather Event

This is the central logic.

The system asks several questions:

1. Is the value physically impossible or structurally invalid?
2. Is it unusual compared with the station’s own recent history?
3. Do temperature, pressure, and humidity behave consistently together?
4. Are nearby stations showing the same change?
5. Is the anomaly persistent or isolated?

Examples:

- If ST001 alone spikes while nearby stations remain normal, the system gives stronger evidence for a likely sensor/data fault.
- If ST001, ST002, ST003, and nearby stations all rise or fall together, the system gives stronger evidence for a possible regional event.

This distinction is handled by combining evidence, not by automatic rule-based guessing.

---

## 9. What Makes the Approach Useful

The approach is useful because it combines:

- rule-based quality checks,
- temporal context,
- multivariate behavior,
- spatial neighbor comparison,
- evidence fusion,
- explainability,
- and sensor health tracking.

This is stronger than a single threshold alarm, and it is much more realistic for a student SIH team to build and demonstrate.

The main value is not just detection, but decision support with evidence.

---

### How We Prove the System Works

We do not simply show a dashboard and claim the system works.

The project demonstrates measurable performance by:

1. replaying real historical weather observations,
2. introducing controlled fault scenarios with known ground truth,
3. running the detection pipeline,
4. comparing the predicted result against the known truth,
5. and using supporting evidence where naturally occurring events are present.

This makes the project defensible and measurable.

For naturally occurring events, the system uses available quality information and contextual evidence to distinguish a genuine regional weather event from an isolated sensor/data failure. We do not automatically label unusual real-world weather as a sensor fault.

---

## 10. MVP Scope

The MVP is intentionally focused on a realistic student project.

The MVP includes:

- real AWS observation data ingestion,
- validation and preprocessing,
- historical and replayed time-series analysis,
- rule-based anomaly checks,
- one practical ML anomaly detector,
- neighbor/spatial comparison for context,
- anomaly explanation,
- sensor health tracking,
- dashboard visibility,
- Docker-based local demo execution.

This is sufficient for a strong SIH prototype and demo.

---

## 11. Features Intentionally Out of Scope

The following are intentionally not part of the MVP:

- automated physical field inspection of AWS hardware,
- a separate model per station,
- full-scale distributed streaming architecture,
- Kafka-based event pipelines,
- Kubernetes deployment,
- large-scale enterprise operations dashboards,
- exact physical cause identification of sensor failures,
- fully autonomous repair workflows,
- generalized global weather forecasting engine.

These items may be useful later, but they are not required for the SIH demo.

---

## 12. Expected Demo Flow

The demo should show the system working on real or replayed weather data.

Expected flow:

1. Open dashboard with network overview.
2. Select a station.
3. Show normal observations for Temperature, Pressure, and Humidity.
4. Start historical replay or the anomaly replay mode.
5. Inject a controlled anomaly at one station.
6. The system detects the suspicious observation.
7. Show evidence such as station history, multivariate inconsistency, and neighbor comparison.
8. Display probable root cause and reasoning.
9. Show the sensor health degradation trend.
10. Compare with surrounding stations to show a true regional event is not incorrectly treated as isolated sensor failure.

The demonstration should be easy to explain and visually understandable.

---

## 13. Important Judge Questions and Concise Answers

### Q: Why is this not just a simple threshold system?
A: Because genuine regional weather changes can also look unusual. The system combines temporal, multivariate, and spatial evidence before deciding.

### Q: Can you really tell whether the sensor is faulty?
A: We can provide a likely or probable diagnosis, not exact physical proof. The system estimates probable root cause with confidence and evidence.

### Q: Why not build one model for each station?
A: That adds complexity without improving the core MVP. A shared detection pipeline is simpler and more maintainable.

### Q: Is the system using real data?
A: Yes. The project is based on real weather observations, with controlled injected anomalies used only for evaluation and demo purposes.

### Q: How do you avoid false alarms?
A: By combining station history, multivariate checks, neighboring station comparison, and a final decision layer.

### Q: Is this feasible for a student team?
A: Yes. This is a modular, practical architecture that can be built and demonstrated locally with Docker and real historical data.

---

## Final Mental Model

The project is:

> AWS Anomaly Intelligence monitors multiple weather stations, detects suspicious observations, compares them with historical behavior and neighboring stations, decides whether the pattern is likely a sensor/data issue or a real weather event, explains the result, and tracks sensor health over time.

Therefore:

**Spatial disagreement can support a sensor-fault hypothesis, while spatial agreement can support a regional-event hypothesis.**

It is evidence, not absolute proof.

---

9. Layer 5 — Evidence Fusion

This is one of the central parts of our design.

Instead of:

One model → Final answer

we use:

Rule score
     +
Temporal score
     +
Multivariate score
     +
Spatial score
     ↓
Evidence Fusion
     ↓
Final anomaly confidence

For example:

Rule-based evidence:      0.90
Temporal evidence:        0.95
Multivariate evidence:    0.82
Spatial evidence:         0.93

Final confidence:         High

The exact fusion method and weights must be determined through validation experiments.

We should not invent weights merely to make the demo look good.

---

10. Layer 6 — Root-Cause Diagnosis

After detecting an anomaly, the system should attempt to classify the probable fault type.

Possible categories:

Normal
Spike
Flatline / Frozen Sensor
Drift
Missing / Communication Failure
Multivariate Inconsistency
Possible Sensor Fault
Possible Regional Event

Example reasoning:

Sudden temperature spike
        +
Nearby stations normal
        +
Pressure and humidity normal
        +
Spike disappears immediately
        ↓
Probable temperature sensor fault

The system should present this as a **probable cause**, not guaranteed physical truth.

---

11. Layer 7 — Sensor Health

The system should maintain a longer-term health state for each station/sensor.

Example:

Health Score
     |
     +--> Healthy
     |
     +--> Warning
     |
     +--> Degraded
     |
     +--> Critical

The score can consider:

recent anomaly frequency

anomaly severity

persistence

affected parameter

historical reliability

This allows the system to support maintenance decisions.

---

12. Layer 8 — Explainability

Every alert should provide understandable evidence.

Example dashboard output:

Station: ST001
Parameter: Temperature

Status:
ANOMALY

Confidence:
96%

Probable Cause:
Temperature Sensor Fault

Why?

✓ Sudden deviation from recent station behavior
✓ Neighboring stations remained stable
✓ Multivariate relationship became inconsistent
✓ Rule-based check was triggered

This is much more useful than:

AI MODEL → ANOMALY

The user should be able to understand why the alert occurred.

---

13. Optional Layer — Corrected / Imputed Values

If an observation is strongly identified as faulty, the system may estimate a plausible replacement value.

Example:

Observed:
55.0°C

Estimated:
30.6°C

However:

**The original raw observation must never be silently overwritten.**

We should preserve:

Raw observation
        +
Derived anomaly label
        +
Optional corrected estimate
        +
Reason/provenance

This keeps the system traceable.

---

14. Real-Time Requirement

The system should support observations arriving continuously.

The production-style flow is:

Data Source
    ↓
Ingestion
    ↓
Preprocessing
    ↓
Quality Checks
    ↓
ML Detection
    ↓
Evidence Fusion
    ↓
Alert
    ↓
Dashboard

For the SIH demonstration, we can simulate this using historical real data.

---

15. Historical Replay for the Demo

Instead of depending on live AWS hardware, we use historical observations and replay them as if they are arriving in real time.

Example:

Historical dataset
       ↓
Replay Engine
       ↓
One record at a time
       ↓
Detection Pipeline
       ↓
Dashboard

This gives us a realistic demonstration while keeping the experiment reproducible.

---

16. Controlled Anomaly Injection

Real datasets usually do not provide perfect labels for every sensor fault.

Therefore, after obtaining real observations, we can create a controlled evaluation dataset.

Example:

Real Data
   ↓
Clean baseline
   ↓
Inject known fault
   ↓
Ground-truth anomaly

Possible injections:

Spike

Normal:    30.5
Injected:  55.0

Flatline

30.1
30.1
30.1
30.1
30.1

Drift

30.1
30.5
30.9
31.3
31.7

Missing data

30.1
30.2
NULL
NULL
30.6

Multivariate corruption

Alter one or more variables so that the combination becomes inconsistent with the learned behavior.

The injection process must be deterministic and documented.

---

17. Evaluation Strategy

We should evaluate the system scientifically.

For anomaly detection:

Precision

Recall

F1-score

False-alarm rate

Detection latency

For root-cause diagnosis:

Root-cause classification accuracy

Confusion matrix

For confidence:

Calibration/reliability of confidence scores

For sensor health:

Ability to identify persistent degradation

Lead time before a maintenance threshold is reached, where ground truth permits

We should compare our complete system against simpler baselines.

---

18. Baselines

The project should not begin with the most complicated model.

We should establish baselines first.

Example progression:

Baseline 1:
Simple rule-based QC

Baseline 2:
Statistical anomaly detection

Baseline 3:
Temporal ML

Baseline 4:
Temporal + Multivariate

Baseline 5:
Temporal + Multivariate + Spatial

Final:
Evidence Fusion + Diagnosis + Health

This lets us answer:

"Does every additional component actually improve the system?"

---

19. Ablation Study

An ablation study means removing one component at a time and measuring the effect.

For example:

Full system
     ↓
Remove spatial module
     ↓
Measure performance

Full system
     ↓
Remove temporal module
     ↓
Measure performance

This provides evidence for why each component exists.

It also prevents us from claiming that a module is useful without testing it.

---

20. Scope of the MVP

The first working version should focus on:

✓ Multiple stations
✓ Temperature
✓ Pressure
✓ Relative Humidity
✓ Historical real data
✓ Basic QC
✓ Temporal detection
✓ Multivariate detection
✓ Spatial comparison where data permits
✓ Evidence fusion
✓ Anomaly confidence
✓ Root-cause categories
✓ Sensor health
✓ Explainable dashboard
✓ Historical replay

Optional features should be added only after the core pipeline works:

○ Corrected/imputed values
○ Advanced predictive maintenance
○ Additional data sources
○ More sophisticated deep-learning architectures

---

21. What Is Outside the Core Scope?

We should not unnecessarily expand the project into:

Building physical AWS hardware

Designing weather sensors

Replacing the official meteorological data infrastructure

Predicting the complete weather forecast

Building a general-purpose disaster prediction system

Using dozens of unrelated weather variables when the PS centers on T/P/RH

Our focus is:

**Observation anomaly detection and sensor/data quality intelligence.**

---

22. Judge-Friendly Explanation

If a judge asks:

"What exactly does your project do?"

Answer:

"Our system monitors observations from multiple automatic weather stations using temperature, atmospheric pressure and relative humidity. It checks each observation against basic quality rules, the station's temporal behavior, multivariate relationships and, where available, neighboring stations. These signals are combined to estimate anomaly confidence and probable cause. The system then explains the evidence and tracks sensor health over time."

---

23. If the Judge Asks: "Why AI/ML?"

Answer:

"Simple rules can identify obvious invalid values, but they are not sufficient for subtle temporal, multivariate and spatial behavior. ML allows the system to learn normal patterns from historical observations and identify deviations that are difficult to capture with fixed thresholds."

---

24. If the Judge Asks: "What Is Novel?"

Do not say:

"We invented anomaly detection."

Instead say:

"The individual techniques are established in research. Our engineering contribution is integrating complementary quality-control, temporal, multivariate and spatial evidence into a real-time AWS intelligence pipeline tailored to the three-parameter constraint, then connecting anomaly detection with interpretable diagnosis and sensor-health tracking. We validate the contribution using real historical observations, controlled anomaly injection, baselines and ablation experiments."

---

25. If the Judge Asks: "Why Not Just Use LSTM?"

Answer:

"A temporal model can detect unusual sequences, but a rapid weather change can be genuine. Temporal behavior alone may therefore produce false alarms. That is why we combine temporal evidence with physical quality checks, multivariate consistency and spatial context."

---

26. If the Judge Asks: "Why Multiple Stations?"

Answer:

"A single station cannot always tell us whether an unusual observation is local or regional. If one station behaves differently while nearby stations remain stable, that supports a station-specific fault hypothesis. If several nearby stations change coherently, the event may be genuine."

---

27. If the Judge Asks: "Do You Train a Model for Every Station?"

Answer:

"No. We can train a common model or common detection pipeline using data from multiple stations. Station-specific historical statistics can be incorporated where useful, while spatial relationships are evaluated during inference."

---

28. If the Judge Asks: "Where Do Your Labels Come From?"

Answer:

"We use real historical observations as the foundation and create controlled anomaly-injected evaluation samples with known ground truth. We also use naturally flagged or documented anomalies where reliable labels are available."

We should never claim that every real-world observation has a verified fault label unless we actually have such documentation.

---

29. If the Judge Asks: "Is Your Data Real?"

Answer:

"Yes. The backbone of our training and evaluation workflow is real historical station observations. Controlled anomalies are injected only for reproducible testing and demonstration, and the original observations are preserved separately."

---

30. If the Judge Asks: "How Do You Prevent False Alarms?"

Answer:

"We do not rely on a single threshold or a single model. We combine multiple evidence sources and explicitly test false-alarm behavior, especially for coherent regional changes where multiple stations behave similarly."

---

31. Final Solution Definition

The final project can be summarized as:

REAL MULTI-STATION WEATHER DATA
              ↓
        DATA QUALITY
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
 TEMPORAL  MULTIVARIATE SPATIAL
    ↓         ↓         ↓
    └─────────┼─────────┘
              ↓
       EVIDENCE FUSION
              ↓
      ANOMALY CONFIDENCE
              ↓
       ROOT-CAUSE HINT
              ↓
        SENSOR HEALTH
              ↓
     EXPLAINABLE ALERT
              ↓
          DASHBOARD

The central principle is:

**Detect → Verify → Explain → Diagnose → Track**

That is the complete problem-and-solution mental model for SIH26073.