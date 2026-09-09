# AWS Anomaly Intelligence

## ML Pipeline and Experiments

This document defines the practical machine-learning approach for the MVP. The goal is a reliable, explainable, demonstrable multi-evidence system that works on real weather data and supports a strong SIH demo.

The implementation stack is Python 3.12 with Pandas, NumPy, SciPy, scikit-learn, PyTorch, PyArrow, and SHAP. The ML pipeline integrates with FastAPI, PostgreSQL, and the Docker-based application; frontend technologies are documented in the application guide.

---

## 1. Baseline

The baseline should be simple and interpretable.

It includes:

- rule-based validation checks,
- station-specific historical summary statistics,
- robust thresholds for physically impossible values,
- local temporal comparisons,
- basic station-to-station difference checks.

This baseline is important because some anomalies are obvious and can be detected without a deep ML model.

---

## 2. Feature Engineering

The model should use engineered features rather than raw values alone.

Useful features include:

- current temperature, pressure, humidity,
- rolling mean of each variable,
- rolling standard deviation,
- z-score relative to the station history,
- time-of-day and day-of-week context,
- difference from previous observation,
- rate of change over a short window,
- multivariate consistency score,
- neighbor difference score,
- station-specific baseline residual.

The feature set should remain compact and interpretable.

---

## 3. Rule-Based QC

Before the main ML detector runs, a rule-based quality-control layer should flag clearly invalid or suspicious records.

Examples:

- missing values,
- impossible ranges,
- duplicate timestamps,
- sudden extreme jumps,
- frozen values over a time window,
- strong multivariate inconsistency.

These checks provide a robust first-pass filter and are helpful for explainability.

---

## 4. Main Anomaly Detection Model

The main detector should be a practical shared model that works across stations rather than a separate model per station.

### Option A: Statistical method

Pros:

- simple,
- transparent,
- low compute cost,
- easy to explain,
- strong baseline.

Cons:

- may not detect more subtle anomalies,
- sensitive to nonstationary weather patterns,
- can produce false alarms without context.

### Option B: Isolation Forest

Pros:

- practical for unsupervised anomaly detection,
- works well on tabular feature vectors,
- fast to train and score,
- relatively explainable compared with deep models,
- usable for station-level and feature-based anomalies.

Cons:

- less explicit about temporal sequences,
- may produce false positives without spatial context,
- does not directly model time dependencies.

### Option C: Autoencoder

Pros:

- captures complex nonlinear structure,
- useful for reconstruction error-based anomaly detection.

Cons:

- more tuning,
- training can be less stable,
- harder to explain,
- may be less reliable for a short student demo without careful preprocessing.

### Option D: GRU/LSTM temporal autoencoder

Pros:

- suitable for time-series learning,
- can learn short temporal dynamics.

Cons:

- more complex,
- requires more data and more engineering,
- harder to debug,
- slower to run,
- more difficult to explain in real-time demo conditions.

---

## 5. Final Multi-Evidence Approach for the MVP

The system is not a single anomaly classifier. Its evidence sources are:

- rule-based meteorological QC,
- Isolation Forest on engineered station-level and temporal features,
- GRU/LSTM-based temporal anomaly detection experiments,
- PCA/Mahalanobis multivariate consistency detection,
- spatial/neighbor comparison when valid station coverage exists.

Evidence fusion combines these signals into anomaly confidence. Isolation Forest is the practical tabular baseline, while GRU/LSTM and PCA/Mahalanobis are evaluated as complementary evidence sources rather than assumed improvements.

This is the most practical balance for the project because it is:

- realistic with available data,
- simpler to implement,
- easier to explain,
- reliable for a demo,
- and computationally lightweight.

The system should not rely only on Isolation Forest as a standalone decision-maker. It should use it along with:

- temporal baseline checks,
- multivariate consistency rules,
- neighbor comparison,
- and evidence fusion.

This makes the final decision stronger and more explainable.

---

## 6. Multivariate Analysis

Temperature, pressure, and humidity should not be treated as completely independent signals.

The system should evaluate whether the combination of variables remains physically and statistically consistent.

Examples of multivariate checks:

- temperature anomaly with no corresponding pressure or humidity change,
- pressure and humidity moving in a way that is inconsistent with local weather patterns,
- temporally consistent but physically implausible variable combinations.

This stage should use PCA and/or Mahalanobis distance over validated multivariate features. It should remain interpretable and should not be treated as proof of a physical fault.

---

## 7. Neighbor/Spatial Evidence

The neighborhood check is critical for reducing false alarms.

For a station at a given time, compute:

- difference from nearby stations,
- shared regional trend score,
- similarity to neighboring stations in the same time window,
- whether the anomaly is isolated or shared.

The spatial model must not use a hard rule such as “X out of Y stations agree.” It calculates a continuous score from usable-neighbor coverage, distance and neighborhood weighting, direction and magnitude similarity, temporal onset/duration alignment, station data quality, station reliability, and elevation/context where available. Track `total_neighbors`, `usable_neighbors`, and `missing_neighbors` separately.

Two nearby stations can support a localized event when their direction, magnitude, timing, and quality are coherent. Two distant stations should produce weak or inconclusive evidence even if their values are similar. A station with poor historical reliability contributes less to consensus.

Common-mode checks cover exact value duplication, identical sequences, suspiciously perfect correlation, stale last-known values, ingestion duplication, and simultaneous source/message failure. This prevents broad agreement from being treated automatically as a genuine regional event.

This helps answer:

- Is the station different from neighbors?
- Or is the entire region changing together?

This is a major part of the project’s value proposition.

---

## 8. Evidence Fusion

The final decision should fuse multiple evidence components:

- rule-based QC score,
- temporal anomaly score,
- multivariate anomaly score,
- spatial/neighbor anomaly score,
- persistence score,
- and severity or confidence estimate.

A simple weighted fusion rule is enough for the MVP:

final_score = w1 * QC + w2 * temporal + w3 * multivariate + w4 * spatial + w5 * persistence

The exact weights should be chosen based on experiments and validation, not guessed in advance.

The system does not assume that consensus automatically means a genuine meteorological event. Spatial consensus is one evidence source and is validated against temporal, multivariate, data-quality and common-mode evidence.

---

## 9. Root-Cause Classification

Root-cause classification should not claim exact physical proof. It should produce a probable explanation such as:

- likely sensor issue,
- possible instrument drift,
- likely data transmission or formatting problem,
- possible regional event,
- or insufficient evidence.

The classifier also supports localized/sub-regional meteorological events, common-mode data faults, stuck sensors, communication faults, sensor drift, and explicit inconclusive or insufficient-spatial-evidence states. A 2-out-of-5 pattern is not classified from the count alone: geographic coherence, temporal alignment, magnitude/direction similarity, reliability, multivariate evidence, and data quality determine whether it is localized, faulty, or inconclusive.

This classification should be based on:

- affected variable,
- persistence,
- severity,
- station history,
- neighborhood agreement,
- and multivariate consistency.

---

## 10. Sensor Health Score

Sensor health should be tracked over time as a rolling metric.

A simple score can combine:

- anomaly frequency,
- anomaly severity,
- persistence,
- variable-specific faults,
- recent behavior.

Example health states:

- healthy,
- watch,
- degraded,
- critical.

This produces a practical maintenance indicator without requiring deep predictive maintenance logic.

---

## 11. Optional Imputation

Imputation is optional and should be treated carefully.

Only use corrected or imputed values when:

- they are clearly marked as derived,
- they are not mistaken for raw observations,
- and they are used only for model support or dashboard smoothing.

The system must never overwrite the original raw data. Imputed values should always be clearly labeled as corrected or estimated values.

## 11. Predictive Maintenance Recommendation

Sensor health is converted into a maintenance recommendation using anomaly frequency, severity, persistence, affected variable, confidence, and recent station behavior. The output may recommend monitoring, review, calibration, or inspection. It is a risk-based recommendation, not an exact physical diagnosis or autonomous repair action.

## 12. Grounded GenAI Layer

GenAI receives structured outputs from evidence fusion, root-cause diagnosis, sensor health, maintenance recommendation, and provenance. It can explain an anomaly, summarize station or network behavior, and answer natural-language investigation questions.

GenAI must not perform primary numerical detection or invent readings, anomalies, confidence scores, station behavior, or evidence. Every response must be traceable to structured pipeline results.

---

## Ground Truth and Evaluation Strategy

Our system must be evaluated against known or established ground truth, not only against a dashboard display. The project is built around measurable evaluation.

### A. Controlled Anomaly Ground Truth

The primary evaluation path starts with real historical weather observations.

The team selects clean/normal observations from historical data and injects controlled anomalies such as:

- sudden temperature spike,
- sudden pressure spike or drop,
- humidity spike or drop,
- flatline or frozen sensor,
- gradual sensor drift,
- missing or communication gap,
- inconsistent multivariate combination,
- isolated station corruption.

Because the team creates the anomaly, the exact ground truth is known:

- when it started,
- when it ended,
- which station was affected,
- which parameter was affected,
- anomaly type,
- original value,
- modified value,
- and the intended label.

Example:

Real temperature:
24.3°C

Injected faulty observation:
48.7°C

Ground truth:

- anomaly = true
- parameter = temperature
- type = spike

Model prediction:

- anomaly = true
- confidence = 0.96

The prediction is then compared directly against the known injected label and metadata. This is the main quantitative evaluation path for the project.

### B. Natural/Real-World Events

The system must also be evaluated against naturally occurring situations where reliable labels or external evidence exist.

This is necessary because not every unusual change in historical data is necessarily a sensor fault.

The project must distinguish:

- genuine regional weather event
from
- isolated sensor/data failure

For example, if temperature changes sharply at many nearby stations at approximately the same time, that may represent a genuine regional weather event rather than a faulty sensor.

Natural unusual weather is not automatically a sensor fault.

This means the evaluation must not assume that every unusual observation in a real dataset is an anomaly. Instead, the system should use station history, multivariate consistency, and multi-station evidence to separate plausible real events from isolated faults.

---

## Evaluation Pipeline

The evaluation flow is conceptually:

```text
REAL HISTORICAL DATA
        |
        +----------------------+
        |                      |
   Clean/normal          Controlled anomaly
        |                      |
        |                Known ground truth
        |                      |
        +-----------> OUR SYSTEM
                           |
                           v
                    Model prediction
                           |
                           v
                  Compare with truth
                           |
              +------------+-------------+
              |            |             |
           Precision     Recall         F1
              |
        False Alarm Rate
        Detection Latency
        Root-Cause Accuracy
        Confidence/Calibration
```

This evaluation should not focus only on whether an anomaly was detected. It should also measure:

1. Detection accuracy
2. False positives / false alarm rate
3. False negatives
4. Detection latency
5. Root-cause classification accuracy
6. Confidence quality / calibration
7. Sensor-health prediction where implemented
8. Correct distinction between isolated sensor faults and regional events

The project should therefore be evaluated on both detection quality and decision quality.

---

## Anomaly Injection Requirements

Anomaly injection must be reproducible and well-documented.

Every injected anomaly should have metadata such as:

- station_id
- timestamp
- parameter
- anomaly_type
- original_value
- modified_value
- severity
- start_time
- end_time
- injection_method
- ground_truth_label

This metadata allows the team to compare the model prediction with known labels and reproduce the experiment exactly.

The project must never overwrite the original dataset.

The system should keep separate data stores for:

- raw data
- cleaned data
- anomaly-injected data
- ground-truth labels
- model predictions

The evaluation set must include spatial edge cases: 5/5 and 4/5 coherent regional changes, 3/5 agreement, nearby versus distant 2/5 agreement, isolated spikes, missing neighbors, unreliable stations, common-mode duplication, flatlines, drift, communication gaps, intermittent faults, multivariate spikes, timing differences, and regional events with different magnitudes.

This separation is essential for reproducible experiments and honest evaluation.

---

## Baseline Comparison

The experiments must compare the proposed system against simpler baselines. The complex model should not be assumed to be better automatically.

### Baseline 1: Rule-based QC

This baseline includes:

- physical/range checks,
- rate-of-change checks,
- persistence/flatline checks,
- missing-value checks.

### Baseline 2: Simple statistical/ML detector

This can include a practical method such as Isolation Forest or another simple detector using engineered features.

### Proposed system

The proposed system is the multi-evidence pipeline combining rule-based QC, Isolation Forest, GRU/LSTM temporal detection, PCA/Mahalanobis multivariate detection, and spatial evidence where valid, followed by evidence fusion, diagnosis, health, maintenance recommendation, and grounded explanation.

The key requirement is that the experiments must show whether each additional component actually improves performance.

---

## Ablation Study

A practical ablation experiment is sufficient for this project.

Example configuration:

A. Rules only
B. Rules + ML
C. Rules + ML + multivariate consistency
D. Rules + ML + multivariate + spatial evidence
E. Full multi-evidence system + diagnosis/health/maintenance layer
F. Full structured system + grounded GenAI explanation layer

Compare each stage on:

- Precision
- Recall
- F1
- False alarm rate
- Detection latency

The goal is to verify which components actually contribute value, rather than assuming the full system is automatically superior.

This is feasible for an SIH project and provides clear evidence for the final demo.

---

## 12. Experiment Design

The project should use a simple and transparent experiment structure.

### Experiment 1: Baseline

- Rule-based validation + basic thresholds only
- Measure how much of the problem is solved by basic QC

### Experiment 2: Baseline + ML

- Add Isolation Forest and compare complementary GRU/LSTM and PCA/Mahalanobis evidence sources
- Measure improvement in detection quality

### Experiment 3: Baseline + ML + Spatial Context

- Add neighbor-based context and evidence fusion
- Test whether spatial information reduces false alarms and improves regional discrimination

### Experiment 4: Full multi-evidence system

- Add multivariate checks, root-cause logic, sensor health, and maintenance recommendation
- Evaluate end-to-end performance and explainability

### Experiment 5: Grounded GenAI evaluation

- Provide GenAI only with structured pipeline results.
- Check that explanations preserve station IDs, timestamps, confidence, evidence, and root-cause labels.
- Check that unsupported questions produce an explicit insufficient-evidence response rather than invented facts.

This progression is enough to show the value of the full system while keeping the project realistic.

---

## 13. Metrics

The system should track the following metrics:

- Precision: fraction of flagged anomalies that are truly anomalous
- Recall: fraction of true anomalies successfully detected
- F1: harmonic mean of precision and recall
- False Alarm Rate: proportion of normal observations incorrectly flagged
- Detection Latency: time between the true anomaly and the system alert
- Root-Cause Accuracy: how often the probable diagnosis matches the known label or supporting evidence
- Confidence Calibration: whether the model confidence is meaningful and ordered correctly

These metrics are practical and explainable to judges.

---

## 14. Controlled Anomaly Injection and Ground Truth

The project should inject anomalies into real data for evaluation and demo purposes.

The injection process should:

- use real historical observations,
- modify only a controlled subset of timestamps,
- record the exact anomaly pattern,
- keep the original data untouched,
- and store the injected anomaly labels separately.

Ground truth is known because the team creates the anomaly pattern deliberately.

This process allows clean evaluation without fabricating the full dataset.

---

## 15. Ablation Experiment: Does Spatial Context Reduce False Alarms?

This is a required experiment.

### Procedure

- Train and evaluate the model without spatial context.
- Repeat with spatial context enabled.
- Compare false alarm rate, precision, and recall.

### Expected outcome

Spatial context should reduce false alarms when a station’s value is unusual but nearby stations show a similar change.

This is one of the strongest arguments for the project’s design.

---

## 16. Avoiding Data Leakage

The project must avoid leakage between training and testing data.

Rules:

- no observations from the test window appearing in the training window,
- no station-level statistics computed from the future being used in the past,
- no injected anomalies appearing in training if the goal is to evaluate normal behavior,
- any feature engineering must be fitted only on the training portion,
- evaluation should be performed on a separate held-out period.

This keeps the results honest and technically credible.

---

## 17. Practical Implementation Guidance

The team should build the pipeline in this order:

1. data validation,
2. baseline statistics,
3. feature engineering,
4. anomaly detector,
5. neighbor comparison,
6. evidence fusion,
7. root-cause classification,
8. sensor health tracking,
9. evaluation on injected anomalies.

This ordering keeps development realistic and reduces confusion between components.

---

## 18. Final Recommendation

For the SIH MVP, the best approach is a compact, explainable multi-evidence pipeline:

- baseline rule checks,
- feature engineering,
- Isolation Forest as a tabular baseline,
- GRU/LSTM temporal evidence,
- PCA/Mahalanobis multivariate evidence,
- neighbor-based contextual reasoning where valid,
- evidence fusion, diagnosis, health, and maintenance recommendation,
- grounded GenAI explanation,
- and evaluation with controlled anomaly injection.

This is the right balance of practicality, reliability, and demo quality.

The project must avoid leakage between training and testing data.

Rules:

- no observations from the test window appearing in the training window,
- no station-level statistics computed from the future being used in the past,
- no injected anomalies appearing in training if the goal is to evaluate normal behavior,
- any feature engineering must be fitted only on the training portion,
- evaluation should be performed on a separate held-out period.

This keeps the results honest and technically credible.

---

## 17. Practical Implementation Guidance

The team should build the pipeline in this order:

1. data validation,
2. baseline statistics,
3. feature engineering,
4. anomaly detector,
5. neighbor comparison,
6. evidence fusion,
7. root-cause classification,
8. sensor health tracking,
9. evaluation on injected anomalies.

This ordering keeps development realistic and reduces confusion between components.

---

## 18. Final Recommendation

For the SIH MVP, the best approach is a compact, explainable multi-evidence pipeline:

- baseline rule checks,
- feature engineering,
- Isolation Forest as a tabular baseline,
- GRU/LSTM temporal evidence,
- PCA/Mahalanobis multivariate evidence,
- neighbor-based contextual reasoning where valid,
- evidence fusion, diagnosis, health, and maintenance recommendation,
- grounded GenAI explanation,
- and evaluation with controlled anomaly injection.

This is the right balance of practicality, reliability, and demo quality.
