# Contributing

## Before Starting

Read the project documentation in `docs/`, especially the project brain, architecture, data validation, and ML pipeline documents. Keep raw data immutable and preserve the separation between processed data, injected anomalies, and ground truth.

## Scope Rules

- Keep changes focused on the owning area of the pipeline.
- Support multiple stations; do not introduce single-station assumptions.
- Do not add datasets, credentials, generated artifacts, or frontend dependencies to source control.
- Update the relevant documentation when an agreed project decision changes.

## Validation

Run the focused tests and checks relevant to a change. The project uses pytest and HTTPX for backend and inference testing once those implementations exist.

## Pull Requests

Describe the purpose of the change, the affected project area, validation performed, and any data or configuration assumptions. Do not commit or push changes without the repository owner's approval.
