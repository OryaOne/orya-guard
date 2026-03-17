# orya-guard
[![CI](https://github.com/OryaOne/orya-guard/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OryaOne/orya-guard/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)


**Preflight validation for ML datasets and inference payloads.**

`orya-guard` is a lightweight Python CLI for catching data problems early, before they break model training, degrade rollout quality, or cause serving-time surprises.

It is intentionally focused: load tabular data, inspect it, compare it, validate inference payloads, and report issues clearly.

## 🧭 Overview

`orya-guard` is built for a narrow but high-leverage part of the ML workflow:

- checking a dataset before training
- comparing a candidate dataset to a trusted reference
- validating inference payloads before serving

The goal is not to replace a broader data platform. The goal is to give engineers a fast, trustworthy preflight check that is easy to run locally, in CI, or in lightweight automation.

## 💡 Why This Exists

Most ML failures do not start with a model.

They start with data that looks almost right:

- a candidate dataset is missing a column
- nulls quietly spike in a critical feature
- a categorical value appears that training never saw
- an inference payload ships with the wrong shape or field types

These issues are common, expensive, and often preventable. `orya-guard` exists to make that preflight check fast, readable, and easy to adopt in local workflows and CI.

## ✨ Features

- **Dataset inspection**: row counts, column counts, dtypes, duplicate rows, null ratios, and constant columns
- **Dataset comparison**: schema mismatches, missing or extra columns, unseen categorical values, null spikes, and simple numeric drift signals
- **Inference validation**: required fields, unexpected fields, and simple type checks against a small JSON schema
- **Clean CLI output**: concise summaries with actionable next steps
- **Optional reports**: write Markdown output for review, CI artifacts, or sharing
- **Open-source ready**: examples, tests, linting, packaging, and GitHub Actions CI included

## 🔎 What It Catches

`orya-guard` currently helps detect:

- schema mismatches
- missing columns
- extra columns
- duplicate rows
- null ratio spikes
- constant columns
- unseen categorical values
- suspicious numeric distribution shifts using simple heuristics
- malformed inference payloads
- missing required fields
- unexpected fields
- simple type mismatches

## 🎯 What It Does Not Do

To keep V1 focused, `orya-guard` does not try to be:

- a data catalog
- a feature store
- a dashboard or web app
- a full data quality platform
- an orchestration system
- an observability product
- a model registry integration layer

It is a sharp preflight tool, not a full MLOps suite.

## 🚀 5-Minute Quickstart

From clone to first useful signal in a few commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

orya-guard check-dataset examples/dataset_with_issues.csv
orya-guard compare examples/train.csv examples/candidate.csv
orya-guard check-inference-payload examples/inference_payload_valid.json --schema examples/inference_schema.json
```

These exact files are documented in [`examples/README.md`](./examples/README.md).

What this gives you immediately:

- a dataset summary with warnings for duplicates, nulls, and constant columns
- a comparison summary with warnings for unseen values, null spikes, and numeric drift
- a passing inference payload validation summary

Run the invalid payload example to see failure behavior:

```bash
orya-guard check-inference-payload examples/inference_payload_invalid.json --schema examples/inference_schema.json
```

That command exits with code `1` and reports missing fields, unexpected fields, and type mismatches.

## 🧪 Examples

### Example files at a glance

- `examples/dataset_with_issues.csv`: small dataset with duplicate rows, nulls, and constant columns
- `examples/train.csv`: reference dataset for comparison
- `examples/candidate.csv`: candidate dataset with null spikes, unseen values, and numeric drift
- `examples/inference_schema.json`: simple schema for payload validation
- `examples/inference_payload_valid.json`: passing payload example
- `examples/inference_payload_invalid.json`: failing payload example

### Check a dataset

```bash
orya-guard check-dataset examples/dataset_with_issues.csv
```

Expected outcome:

- `Status: WARN`
- duplicate rows detected
- `score` reported with a `50.0%` null ratio
- constant columns reported

Example output:

```text
Dataset Check
Status: WARN

Summary
-------
- Dataset: examples/dataset_with_issues.csv
- Rows: 4
- Columns: 4
- Column dtypes: constant_feature=str, customer_id=int64, score=float64, status=str

Key findings
------------
- Duplicate rows: 1
- Columns with nulls: 1
- Constant columns: 2
```

### Compare train and candidate data

```bash
orya-guard compare examples/train.csv examples/candidate.csv
```

Expected outcome:

- `Status: WARN`
- no missing columns
- unseen `city` value `volos`
- `income` null ratio change
- numeric drift warnings

Example output:

```text
Dataset Comparison
Status: WARN

Summary
-------
- Training dataset: examples/train.csv
- Candidate dataset: examples/candidate.csv

Key findings
------------
- Missing columns: 0
- Extra columns: 0
- Incompatible dtypes: 0
- Null ratio changes: 1
- Unseen categorical values: 1
- Numeric drift signals: 2
```

### Validate an inference payload

```bash
orya-guard check-inference-payload examples/inference_payload_valid.json --schema examples/inference_schema.json
```

Expected outcome:

- `Status: PASS`
- no missing required fields
- no unexpected fields
- no type mismatches

Example output:

```text
Inference Payload Check
Status: PASS

Summary
-------
- Payload: examples/inference_payload_valid.json
- Schema: examples/inference_schema.json

Key findings
------------
- Missing required fields: 0
- Unexpected fields: 0
- Type mismatches: 0
```

### Write a Markdown report

```bash
orya-guard check-dataset examples/dataset_with_issues.csv --report reports/dataset-report.md
orya-guard compare examples/train.csv examples/candidate.csv --report reports/compare-report.md
```

Expected outcome:

- the CLI prints the summary
- a Markdown report is written to `reports/`

## 🔁 End-to-End Demo

This is the fastest way to understand the project after cloning the repo.

1. Inspect the candidate dataset before using it in training.
2. Compare it against the reference training dataset.
3. Save a Markdown report for review or CI artifacts.
4. Validate a sample inference payload before shipping the serving contract.

```bash
orya-guard check-dataset examples/candidate.csv
orya-guard compare examples/train.csv examples/candidate.csv --report reports/compare.md
orya-guard check-inference-payload examples/inference_payload_valid.json --schema examples/inference_schema.json
```

What this gives you:

- a quick quality snapshot of the candidate data
- an explicit warning if the serving or training contract drifted
- a shareable comparison report in Markdown
- confidence that an example inference payload matches the declared schema

If you want to see a failure case immediately:

```bash
orya-guard check-inference-payload examples/inference_payload_invalid.json --schema examples/inference_schema.json
```

That command should return `Status: FAIL` and explain the missing field, unexpected field, and type mismatches.

## 🛠️ Command Reference

### `orya-guard check-dataset <path>`

Inspect a CSV or parquet dataset and print a summary.

Checks:

- row count
- column count
- pandas dtypes
- duplicate rows
- null ratios per column
- constant columns

Options:

- `--report`, `-o`: write a Markdown report to a file

Notes:

- exits with code `0` when the file is readable, even if warnings are found

### `orya-guard compare <train_path> <candidate_path>`

Compare a candidate dataset against a training reference dataset.

Checks:

- missing columns
- extra columns
- dtype compatibility mismatches
- null ratio changes
- unseen categorical values
- suspicious numeric drift

Options:

- `--report`, `-o`: write a Markdown report to a file

Notes:

- exits with code `1` when blocking compatibility issues are found, such as missing columns or incompatible dtypes
- exits with code `0` when only warnings are present

### `orya-guard check-inference-payload <json_path> --schema <schema.json>`

Validate one inference payload against a simple JSON schema.

Checks:

- payload must be a JSON object
- missing required fields
- unexpected fields
- simple type validation

Notes:

- exits with code `1` when the payload is invalid

## 📦 Inference Schema Format

`orya-guard` uses a deliberately small schema format for inference validation:

```json
{
  "fields": {
    "customer_id": { "type": "integer", "required": true },
    "age": { "type": "integer", "required": true },
    "city": { "type": "string", "required": true },
    "income": { "type": "number", "required": false },
    "is_premium": { "type": "boolean", "required": false }
  }
}
```

Supported field types:

- `string`
- `integer`
- `number`
- `boolean`
- `object`
- `array`

This format is intentionally simple so teams can adopt it without learning a larger validation system first.

## 🧱 Project Structure

```text
orya_guard/
  checks/
    comparison.py
    dataset.py
    inference.py
  io/
    data_loading.py
  models/
    results.py
    schema.py
  reporting/
    console.py
    markdown.py
  cli.py
examples/
  README.md
tests/
.github/workflows/ci.yml
pyproject.toml
```

Design choices:

- `orya_guard/checks/` contains reusable validation logic, separate from the CLI
- `orya_guard/models/` keeps results explicit and typed with Pydantic
- `orya_guard/reporting/` isolates output formatting from the checks themselves
- `examples/` and `tests/` ensure the quickstart stays grounded in real, runnable behavior

## 🧑‍💻 Local Development

Install development dependencies and run the same checks used in local development and CI:

```bash
pip install -e ".[dev]"
ruff format .
ruff check .
pytest
```

The repository includes:

- `pytest` tests for CLI and core validation logic
- `ruff` for linting
- GitHub Actions CI in [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)

## 🗺️ Roadmap

Near-term improvements that fit the project scope:

- configurable thresholds for null spikes and numeric drift
- optional machine-readable output formats
- richer report formatting
- support for directories or batch validation workflows
- stricter schema options for inference validation

Out of scope for this project direction:

- dashboards
- cloud deployment features
- authentication
- database-backed state
- broad MLOps platform integrations

## 🤝 Contributing

Contributions are welcome and should stay tightly aligned with the project scope.

If you want to help:

1. Open an issue describing the bug, improvement, or proposed check.
2. Keep changes focused on preflight validation for datasets and inference payloads.
3. Add or update tests when behavior changes.
4. Run `ruff format .`, `ruff check .`, and `pytest` before opening a pull request.

Good contributions for this project:

- clearer CLI UX
- better reports
- additional focused checks
- better docs and examples
- test coverage improvements

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE).
