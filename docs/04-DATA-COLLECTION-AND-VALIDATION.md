# AWS Anomaly Intelligence

## Data Collection and Validation SOP

This document is the data team’s operating guide for collecting, validating, and accepting AWS datasets for the project. The goal is to use real observational weather data, keep raw data untouched, and build a clean pipeline for experimentation and demo work.

Data is exchanged as CSV, Parquet, and JSON files and stored in PostgreSQL when application persistence is required. Python 3.12 with Pandas, NumPy, SciPy, PyArrow, and Pydantic will support collection, validation, and schema handling.

---

## 1. What Data We Need

The project uses observations from multiple Automatic Weather Stations (AWS) with the following essential variables:

- timestamp
- station_id
- temperature
- atmospheric pressure
- relative humidity

We also need the following station metadata for each station where available:

- latitude
- longitude
- elevation
- station name or identifier

The project is not building hardware. It is building the software layer that analyzes the observations produced by AWS stations.

---

## 2. Required Variables

The minimum required set is:

- timestamp
- station_id
- temperature
- pressure
- relative_humidity

The recommended station metadata fields are:

- latitude
- longitude
- elevation
- station name/identifier

These values are necessary for:

- temporal anomaly detection,
- multivariate checks,
- spatial neighbor comparisons,
- evidence explanation,
- sensor-health tracking,
- and maintenance recommendations.

Spatial reasoning is conditional. Neighbor comparisons are used only when station coordinates, timestamps, variable coverage, and sufficient nearby observations are valid. The system must report insufficient coverage rather than claim a regional event without support.

---

## 3. Recommended Real Data Sources

The project should prioritize real observational weather data over model-derived or synthetic data.

The following sources are reasonable candidates to evaluate. Each must be checked for actual fields before acceptance.

### 3.1 NOAA NCEI Climate and Weather Data

- Source name: NOAA National Centers for Environmental Information (NCEI)
- Official URL: https://www.ncei.noaa.gov/
- What data it provides: Observed surface weather and climate records from stations and datasets
- Temperature: Yes, often provided as station observations
- Pressure: Yes, often provided in some datasets
- Relative humidity: Yes, often included in station observations
- Temporal resolution: Hourly or daily depending on dataset
- Station metadata: Yes, station coordinates and metadata are usually available
- Download method: API and bulk download from NCEI portals
- Important limitations: Some datasets may be partial, may have different station coverage by region, and may require filtering for the exact time window
- Observed or modeled: Usually observed, but a teammate must verify the exact dataset and fields before use

### 3.2 India Meteorological Department (IMD) AWS or surface data

- Source name: India Meteorological Department (IMD)
- Official URL: TBD — to be finalized during implementation
- What data it provides: Surface weather observations from AWS and meteorological stations in India
- Temperature: Yes
- Pressure: Yes
- Relative humidity: Yes
- Temporal resolution: Depends on the dataset; often hourly or sub-daily
- Station metadata: Usually yes, including station location and identifiers
- Download method: Institutional/public data access depending on the dataset and policy
- Important limitations: Access policies and file formats may differ; validation is required
- Observed or modeled: Observed, but field availability must be verified

### 3.3 Meteostat

- Source name: Meteostat
- Official URL: https://meteostat.net/
- What data it provides: Historical weather observations from meteorological stations
- Temperature: Yes
- Pressure: Yes
- Relative humidity: Yes
- Temporal resolution: Hourly or daily depending on station
- Station metadata: Yes
- Download method: API and bulk download
- Important limitations: Dataset availability depends on station coverage; not every station includes all variables in the same way
- Observed or modeled: Usually observational, but exact station records must be checked

### 3.4 Open-Meteo historical weather API

- Source name: Open-Meteo
- Official URL: https://open-meteo.com/
- What data it provides: Historical weather data and forecasts
- Temperature: Yes
- Pressure: Yes
- Relative humidity: Yes
- Temporal resolution: Hourly or daily
- Station metadata: Indirect via grid points or nearby locations; not necessarily station-based
- Download method: API
- Important limitations: This is not always a true station-level AWS dataset; it may be model-derived or gridded and may not match the exact station-based requirement
- Observed or modeled: Often modeled or gridded; not the preferred source for a station-centric AWS anomaly system unless the exact data source is validated

### 3.5 Local or institutional AWS station archives

- Source name: Project-specific local or institutional AWS archive
- Official URL: TBD — to be finalized during implementation
- What data it provides: Station observations from local AWS networks
- Temperature: Depends on archive
- Pressure: Depends on archive
- Relative humidity: Depends on archive
- Temporal resolution: Depends on archive
- Station metadata: Usually yes
- Download method: Institutional transfer, CSV export, or API
- Important limitations: Quality and consistency vary; all variables must be validated
- Observed or modeled: Usually observed, but must be verified

Important rule: do not accept any dataset until its actual fields, units, and provenance are verified.

---

## 4. Exact Expected Dataset Schema

The canonical table for the processed data should be:

| timestamp | station_id | temperature | pressure | relative_humidity | latitude | longitude | elevation |
| --- | --- | --- | --- | --- | --- | --- | --- |

### 4.1 Field Definitions

- timestamp: observation time in UTC or local time with timezone normalization
- station_id: unique AWS station identifier
- temperature: temperature in °C
- pressure: atmospheric pressure in hPa
- relative_humidity: relative humidity in %
- latitude: decimal degrees
- longitude: decimal degrees
- elevation: meters above sea level

### 4.2 Data Types

- timestamp: datetime or ISO 8601 timestamp
- station_id: string
- temperature: float
- pressure: float
- relative_humidity: float
- latitude: float
- longitude: float
- elevation: float

### 4.3 Timestamp Format

Use a consistent format such as:

- YYYY-MM-DD HH:MM:SS
- or ISO 8601 UTC format such as 2025-01-01T10:15:00Z

The team should choose a single standard and convert all source data accordingly.

### 4.4 Missing Value Representation

Missing values must be stored consistently. Common practices include:

- NULL / NaN in raw source parsing
- or an explicit missing marker during processing

The raw data should remain unchanged even if missing values are handled later during preprocessing.

### 4.5 Duplicate Handling

Duplicate observations should be identified using:

- station_id + timestamp
- and optionally any UNIQUE source record ID if present

The raw dataset should be kept, but duplicate records should be flagged and either removed or reconciled before model training or evaluation.

---

## 5. Data Collection Procedure

The data collection process should be explicit and repeatable.

### Step 1: Identify candidate datasets

- Select one or two real AWS data sources.
- Check whether they provide the required variables and metadata.
- Confirm the station coverage and time range.

### Step 2: Download raw data

- Download the raw observation files in their original form.
- Preserve file metadata, timestamps, and source naming.
- Do not edit source files.

### Step 3: Validate source schema

- Confirm all required columns exist.
- Check temperature, pressure, and humidity units.
- Record how missing values are represented.
- Check whether timestamps are in local time or UTC.

### Step 4: Standardize to the canonical schema

- Rename columns to the project standard names.
- Convert time to one consistent format.
- Convert units to the chosen project units.
- Keep the raw copy and a processed copy separately.

### Step 5: Add station metadata

- Join or map station IDs to latitude, longitude, and elevation where possible.
- Keep a station metadata table if the source data is distributed across multiple files.

### Step 6: Run validation checks

- Check missing values and duplicates.
- Check unrealistic sensor readings.
- Check timestamp continuity and ordering.
- Check the spatial metadata for plausibility.
- Confirm that neighboring-station comparisons are possible before deriving spatial features.

### Step 7: Save raw and processed versions separately

The processed data may be used for analysis, but the raw observations must remain immutable.

---

## 5A. Data Collection for Evaluation and Ground Truth

The dataset collection process must support evaluation, not just model training.

The data collection teammate must provide:

1. Raw historical observations
2. Station metadata
3. Timestamp information
4. Temperature
5. Relative humidity
6. Atmospheric pressure
7. Data source / provenance
8. Quality flags if available
9. Any source-specific observation or status flags
10. Neighboring station information if available

The raw data must never be modified.

The project should clearly separate:

- REAL OBSERVATION
- INJECTED ANOMALY
- MODEL PREDICTION

Controlled anomaly injection should happen only after the real dataset has been collected and validated.

This ensures that synthetic evaluation data is generated from clean historical observations, not by editing the original data source itself.

---

## 6. Data Quality Checks

The dataset must pass each quality check before being accepted.

### 6.1 Missing values

- Count null values by column.
- Identify whether missing data is random or clustered.
- Check whether a missing block is a real outage or a data export issue.

### 6.2 Duplicates

- Detect duplicate station_id + timestamp combinations.
- Flag repeated records.
- Decide whether they are true duplicates or repeated transmissions.

### 6.3 Invalid timestamps

- Check for empty timestamps.
- Validate date ranges.
- Detect impossible times and timezone inconsistencies.

### 6.4 Impossible values

Examples:

- temperature values below -100°C or above 100°C outside a realistic range,
- pressure values below 700 hPa or above 1100 hPa,
- relative humidity values below 0 or above 100.

### 6.5 Unit consistency

- Temperature must be in °C.
- Pressure must be in hPa.
- Relative humidity must be in %.

Do not mix units across datasets without conversion.

### 6.6 Suspicious constant values

- A station may report exactly the same value for many consecutive timestamps.
- This can indicate a frozen sensor or a bad data stream.

### 6.7 Extreme jumps

- Detect sudden spikes between consecutive timestamps.
- Use both a single-record threshold and a broader time-window perspective.

### 6.8 Station metadata validity

- Check latitude and longitude ranges.
- Check elevation plausibility.
- Confirm that station identifiers are consistent across files.

---

## 7. Cross-Check Procedure

The data teammate should verify that data is real and not accidentally synthetic, duplicated, modeled, incorrectly converted, or corrupted.

Checklist:

- Compare raw source values with the processed version.
- Confirm that the timestamp format is consistent.
- Ensure units are converted exactly once and recorded clearly.
- Re-check a random subset of records against the original source file.
- Verify station IDs are not duplicated or merged incorrectly.
- Confirm coordinate metadata matches the expected region.
- Check for repeated records or near-duplicate files.
- Review whether the data came from observed weather stations or a modeled grid.

If the dataset cannot be traced clearly to a real observation source, it should not be accepted.

---

## 8. Dataset Acceptance Checklist

A dataset should only be accepted when all of the following pass:

- required variables are present,
- units are consistent,
- timestamps are valid and normalized,
- station metadata is present or verifiable,
- missing values are documented,
- duplicates are handled,
- impossible values are identified,
- station locations are plausible,
- raw records and processed records are clearly separated,
- data provenance is recorded,
- the source is confirmed as real observational data where possible.

Any dataset failing these checks should be rejected or fixed before use.

---

## 9. Data Folder and Provenance Recommendations

The project should logically separate data into the following categories:

- RAW DATA: original source downloads and untouched records
- PROCESSED DATA: cleaned, normalized, and validated records
- INJECTED TEST DATA: controlled anomalies added for evaluation only
- FEATURE DATA: engineered features used by the ML pipeline
- MODEL EVALUATION DATA: train/validation/test partitions and metrics metadata
- GROUND TRUTH: injection labels, affected stations and variables, timing, severity, and provenance

This separation is important because the raw observation is the ground truth and should never be overwritten.

---

## 10. Anomaly Injection Dataset

Anomaly injection is essential for evaluation and demonstration, but it must be separate from the real data.

### REAL DATA

These are the original weather observations and the baseline dataset used for the project.

### INJECTED TEST DATA

These are deliberately modified copies created only for evaluation. They must not overwrite the original files.

Controlled anomaly types:

- spike
- drop
- pressure anomaly
- humidity anomaly
- flatline
- drift
- missing block
- intermittent fault
- multivariate inconsistency
- local station anomaly
- regional meteorological event
- persistent sensor degradation

Examples:

- Spike: replace a valid temperature reading with an extreme value
- Flatline: freeze a variable for multiple timestamps
- Drift: apply a gradual bias over several observations
- Missing block: set several consecutive observations to missing
- Multivariate inconsistency: alter temperature and humidity together in a way that is inconsistent with the station’s typical behavior

Each injected anomaly should be recorded with:

- station_id,
- timestamp,
- anomaly type,
- injected value range or pattern,
- whether it is for evaluation or demo use,
- and ground-truth label.

The ground-truth record should also preserve the original value, injected value or pattern, severity, start and end timestamps, injection method, and whether the case is intended for evaluation or demo replay. Ground truth is stored separately from injected observations and model predictions.

This makes evaluation reproducible and prevents confusion between real and injected values.

---

## 11. Minimal Data Governance Rule

The project must never confuse:

- REAL OBSERVATION
- PROCESSED OBSERVATION
- DETECTED ANOMALY
- INJECTED ANOMALY
- CORRECTED/IMPUTED VALUE

This distinction is critical for honest evaluation and explainability.

Validated observations, quality flags, evidence scores, and provenance may be passed as structured results to the GenAI explanation layer. GenAI may summarize or explain those results, but it must not modify raw data or invent readings, anomalies, confidence, station behavior, or evidence.
