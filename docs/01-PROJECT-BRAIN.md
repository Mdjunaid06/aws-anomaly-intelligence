AWS Anomaly Intelligence

Project Brain

This document explains the complete idea behind the project. Every team member should read this before working on the project.

---

1. Project Identity

**Project Name:** AWS Anomaly Intelligence

**SIH Problem Statement:** SIH26073

**Problem Statement:** AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

**Final system vision:** AI-powered, explainable, spatiotemporal anomaly intelligence and predictive maintenance platform for Automatic Weather Stations.

**Organization:** Ministry of Earth Sciences (MoES)

**Theme:** Disaster Management

**Category:** Software

---

## Final Technology Stack

The frontend will use React.js, Vite, Tailwind CSS, Recharts, Leaflet, React-Leaflet, and Axios. The backend will use Python 3.12, FastAPI, Uvicorn, Pydantic, SQLAlchemy, PostgreSQL, and python-dotenv. Data and ML work will use Pandas, NumPy, SciPy, scikit-learn, PyTorch, PyArrow, and SHAP, with pytest and HTTPX for testing and Docker Compose for local deployment.

---

2. The Problem in Simple Words

Automatic Weather Stations continuously collect weather observations.

For this problem, we are concerned with only three main measurements:

Temperature

Atmospheric Pressure

Relative Humidity

The problem is that these observations can sometimes be wrong.

For example:

10:00 → 30.1°C
10:05 → 30.3°C
10:10 → 30.5°C
10:15 → 55.0°C
10:20 → 30.7°C

The value `55.0°C` may be a sensor problem.

However, we cannot simply say:

"55°C is abnormal, therefore the sensor is faulty."

Because unusual weather can actually happen.

The real problem is:

**How can we determine whether an unusual observation is a genuine weather event or a faulty sensor/data observation?**

That is what our system is designed to solve.

---

3. What is an Automatic Weather Station?

An Automatic Weather Station (AWS) is a system that automatically measures and transmits weather observations from a particular location.

Conceptually:

                 AWS
                  |
       +----------+----------+
       |          |          |
       ↓          ↓          ↓
 Temperature   Pressure   Humidity
       |          |          |
       +----------+----------+
                  |
                  ↓
            Data Transmission
                  |
                  ↓
             Data System

The physical AWS hardware is **not what we are building**.

We are building the intelligent software layer that analyzes the observations produced by AWS stations.

---

4. What Exactly Are We Building?

We are building an:

**AI-powered, explainable, spatiotemporal AWS anomaly intelligence and predictive maintenance platform.**

The system receives observations from multiple AWS stations and continuously checks whether the observations look normal.

When something suspicious occurs, the system should:

1. Detect the anomaly.

2. Measure how unusual it is.

3. Check historical behavior.

4. Check the relationship between temperature, pressure and humidity.

5. Compare with other relevant stations where spatial information is available.

6. Determine the probable cause.

7. Assign confidence and severity.

8. Track the health of the affected sensor.

9. Explain why the system generated the alert.

10. Recommend maintenance action from sensor-health evidence.

11. Preserve optional traceable corrected or imputed values without changing raw data.

12. Provide a grounded GenAI layer for explanations, summaries, and natural-language investigation.

---

5. The Most Important Concept: Multiple Stations

A major part of our solution is understanding that one station should not always be analyzed in isolation.

Suppose we have:

             ST001
               |
       +-------+-------+
       |       |       |
     ST002   ST003   ST004

All stations contain observations of:

Temperature

Atmospheric Pressure

Relative Humidity

Now imagine:

ST001 → 55°C

ST002 → 31°C
ST003 → 31.2°C
ST004 → 30.9°C

ST001 is behaving very differently from the surrounding stations.

This provides strong evidence that the observation may be faulty.

---

6. But What If All Stations Change?

This is extremely important.

Suppose:

ST001 → 38°C
ST002 → 38.2°C
ST003 → 37.9°C
ST004 → 38.1°C

If all nearby stations change in a similar way, this may be a genuine meteorological event.

Therefore:

Different from neighbors
        ≠
Automatically faulty

Instead, the system combines multiple pieces of evidence.

---

7. Our Core Reasoning

For a suspicious observation, we ask:

Question 1 — Is it physically suspicious?

Example:

Missing value
Impossible value
Very large sudden change

Question 2 — Is it unusual for this station?

Compare it with the station's historical behavior.

Question 3 — Are the three parameters behaving consistently?

Analyze:

Temperature
Pressure
Humidity

together.

Question 4 — Are nearby stations behaving similarly?

If other stations also changed, the event may be genuine.

Question 5 — Is the problem persistent?

A single unusual value and a sensor that has been producing bad values for several hours are different situations.

---

8. The Complete Reasoning Pipeline

                 AWS Observations
                        |
                        ↓
                Data Validation
                        |
                        ↓
              Basic Quality Checks
                        |
            +-----------+-----------+
            |           |           |
            ↓           ↓           ↓
        Temporal    Multivariate   Spatial
        Analysis     Analysis      Analysis
            |           |           |
            +-----------+-----------+
                        |
                        ↓
                 Evidence Fusion
                        |
                        ↓
                Anomaly Decision
                        |
             +----------+----------+
             |                     |
             ↓                     ↓
        Root Cause            Sensor Health
        Analysis                Tracking
             |
             ↓
       Explainable Alert

---

9. What Does "Anomaly Detection" Mean Here?

An anomaly is an observation or sequence of observations that significantly differs from expected behavior.

Possible anomaly types include:

Spike

30.1
30.3
30.4
55.0
30.6

Flatline

30.1
30.1
30.1
30.1
30.1

This can indicate a frozen sensor.

Drift

30.1
30.8
31.5
32.2
33.0

A gradual abnormal change may indicate sensor degradation.

Missing/Communication Failure

30.1
30.2
NULL
NULL
NULL
30.8

Multivariate Inconsistency

Temperature, pressure and humidity may contain a combination that is inconsistent with learned behavior.

Other supported experiment types include temperature drops, pressure or humidity anomalies, intermittent faults, local station anomalies, regional meteorological events, and persistent sensor degradation.

---

10. What Does Our System Actually Return?

The system should not only return:

ANOMALY = YES

Instead, it should produce something like:

Station ID: ST001

Timestamp:
2025-01-01 10:15

Affected Parameter:
Temperature

Status:
Anomaly

Severity:
Critical

Confidence:
96%

Probable Cause:
Temperature Sensor Fault

And importantly:

Evidence:

Temporal behavior       → Strong anomaly
Multivariate behavior   → Strong anomaly
Spatial comparison      → Strong anomaly
Rule-based checks       → Strong anomaly

This makes the decision explainable.

---

11. Root Cause vs Anomaly

These are not the same thing.

Anomaly detection asks:

"Does this observation look abnormal?"

Root-cause analysis asks:

"What type of problem could have caused this abnormal behavior?"

For example:

Anomaly detected
       ↓
Sudden spike
       ↓
Only temperature affected
       ↓
Neighboring stations normal
       ↓
Probable temperature sensor fault

The root cause is a **probable diagnosis**, not guaranteed physical proof.

---

12. Sensor Health

We also want to maintain a longer-term health view of each sensor.

For example:

ST001

Healthy
   ↓
Occasional anomaly
   ↓
Repeated anomalies
   ↓
Degraded
   ↓
Critical

The health score can consider:

frequency of anomalies

severity

persistence

recent history

affected parameter

This allows the system to move from:

"Something is wrong right now."

towards:

"This sensor has been showing repeated abnormal behavior and should be inspected."

---

13. Real Data Strategy

We will use **real historical weather observations** as the foundation of the project.

We do not want to train the entire system using artificial toy data.

The process will be:

Real Historical Weather Data
             |
             ↓
       Data Validation
             |
             ↓
      Clean/Normal Data
             |
             ↓
       Model Training

For evaluation, we can inject controlled faults into real observations:

Real Observation
       +
Controlled Fault
       ↓
Known Anomaly

For example:

Real:
31.2°C

Injected:
55.0°C

Because we know exactly where and when we injected the fault, we have ground truth for evaluation.

---

14. Important: We Do NOT Need One Model Per Station

Suppose we collect:

ST001
ST002
ST003
...
ST020

We do **not** automatically create:

Model_ST001
Model_ST002
Model_ST003
...
Model_ST020

That would unnecessarily complicate the system.

Instead, we can develop a common detection pipeline using data from multiple stations.

Station-specific statistics or calibration can be added where useful.

The spatial comparison happens between stations during inference.

---

15. How the Demo Will Work

We cannot depend on a physical AWS sensor being available during the SIH presentation.

Therefore, the demo will use a **real-data replay system**.

Conceptually:

Historical Real Data
        |
        ↓
   Replay Engine
        |
        ↓
 Observations appear
 as if arriving live
        |
        ↓
       ML System
        |
        ↓
     Dashboard

During the demonstration, we can trigger a controlled fault.

Example:

Normal:

ST001 → 30.2°C
ST002 → 30.4°C
ST003 → 30.1°C


Fault injected:

ST001 → 55.0°C

ST002 → 30.5°C
ST003 → 30.4°C

The system should detect the suspicious ST001 observation.

---

16. The Strongest Demo Scenario

The best demonstration should contain two situations.

Scenario A — Sensor Fault

ST001 → sudden abnormal value

Other stations → normal

Expected:

Anomaly detected
High confidence
Probable sensor fault

Scenario B — Genuine Regional Event

ST001 → changes
ST002 → changes
ST003 → changes
ST004 → changes

Expected:

Regional event likely
Do not incorrectly classify every station as faulty

This demonstrates that our system is doing more than simple threshold detection.

---

17. GenAI Responsibility

GenAI is not the primary anomaly detector. Numerical detection remains the responsibility of rule-based QC, statistical methods, and ML models. GenAI receives structured pipeline results and may explain individual anomalies, summarize station or network behavior, answer natural-language investigation questions, and explain why evidence supports a likely sensor fault or regional event.

GenAI must not invent readings, anomalies, confidence scores, station behavior, or evidence. Its output must be traceable to structured results.

---

18. What Makes the Project Technically Strong?

The strength does not come from saying:

"We use AI."

Instead, the strength comes from combining different evidence sources.

Rule-based QC
      +
Temporal behavior
      +
Multivariate behavior
      +
Spatial behavior
      +
Evidence fusion
      +
Root-cause analysis
      +
Sensor health
      +
Explainability

Each component should be experimentally evaluated.

---

19. What We Must NOT Claim

We must not claim that:

anomaly detection itself is new

LSTM is new

autoencoders are new

SHAP is new

spatial anomaly detection is new

predictive maintenance is new

multivariate weather anomaly detection is new

These techniques already exist in research.

Our contribution is the **integration and validation of a complete real-time AWS anomaly intelligence pipeline under the specific SIH26073 requirements**.

---

20. Project Development Order

The team should build the project in this order:

1. Find real datasets
        ↓
2. Validate the datasets
        ↓
3. Build preprocessing pipeline
        ↓
4. Build rule-based baseline
        ↓
5. Build temporal detector
        ↓
6. Build multivariate detector
        ↓
7. Build spatial detector
        ↓
8. Build evidence fusion
        ↓
9. Build root-cause analysis
        ↓
10. Build sensor-health and maintenance recommendation system
        ↓
11. Build structured results and grounded GenAI explanation layer
        ↓
12. Build backend
        ↓
13. Build frontend
        ↓
14. Build replay/demo system
        ↓
15. Dockerize everything
        ↓
16. Run experiments
        ↓
17. Prepare SIH demo

---

21. Golden Rule of the Project

Every alert should answer:

**"Why did the system think this observation was abnormal?"**

Not simply:

**"The AI says it is abnormal."**

The final system should provide evidence that a human can understand.

---

22. One-Line Mental Model

Remember the project like this:

**"We monitor multiple weather stations, detect and diagnose suspicious observations with spatiotemporal evidence, recommend maintenance from sensor health, and use grounded GenAI to explain the structured decision to humans."**