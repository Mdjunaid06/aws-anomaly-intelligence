# AWS Anomaly Intelligence

## ML Pipeline and Experiments

This document defines the practical machine-learning approach for the MVP. The goal is not to use the most complicated model possible. The goal is a reliable, explainable, demonstrable system that works on real weather data and supports a strong SIH demo.

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

### Option D: LSTM/GRU-based model

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

## 5. Recommended Primary Approach for the MVP

The recommended primary model is:

> Isolation Forest on engineered station-level and temporal features, combined with rule-based QC and spatial context.

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

This stage can use simple residual-based or distance-based methods and should not require a large complex model.

---

## 7. Neighbor/Spatial Evidence

The neighborhood check is critical for reducing false alarms.

For a station at a given time, compute:

- difference from nearby stations,
- shared regional trend score,
- similarity to neighboring stations in the same time window,
- whether the anomaly is isolated or shared.

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

---

## 9. Root-Cause Classification

Root-cause classification should not claim exact physical proof. It should produce a probable explanation such as:

- likely sensor issue,
- possible instrument drift,
- likely data transmission or formatting problem,
- possible regional event,
- or insufficient evidence.

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

---

## 12. Experiment Design

The project should use a simple and transparent experiment structure.

### Experiment 1: Baseline

- Rule-based validation + basic thresholds only
- Measure how much of the problem is solved by basic QC

### Experiment 2: Baseline + ML

- Add the main anomaly detector
- Measure improvement in detection quality

### Experiment 3: Baseline + ML + Spatial Context

- Add neighbor-based context and evidence fusion
- Test whether spatial information reduces false alarms and improves regional discrimination

This progression is enough to show the value of the full system.

---

## 13. Metrics

The system should track the following metrics:

- Precision: fraction of flagged anomalies that are truly anomalous
- Recall: fraction of true anomalies successfully detected
- F1: harmonic mean of precision and recall
- False Alarm Rate: proportion of normal observations incorrectly flagged
- Detection Latency: time between the true anomaly and the system alert

These metrics are practical and easy to explain to judges.

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

For the SIH MVP, the best approach is a compact, explainable pipeline:

- baseline rule checks,
- feature engineering,
- Isolation Forest as the main detector,
- neighbor-based contextual reasoning,
- evidence fusion,
- sensor health tracking,
- and evaluation with controlled anomaly injection.

This is the right balance of practicality, reliability, and demo quality.
