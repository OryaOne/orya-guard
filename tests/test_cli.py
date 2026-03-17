from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from orya_guard.cli import app


def test_check_dataset_cli_writes_markdown_report(
    cli_runner: CliRunner,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "dataset-report.md"

    result = cli_runner.invoke(
        app,
        [
            "check-dataset",
            str(examples_dir / "dataset_with_issues.csv"),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Dataset Check" in result.stdout
    assert "Key findings" in result.stdout
    assert report_path.exists()
    assert "Dataset Check Report" in report_path.read_text(encoding="utf-8")


def test_compare_cli_reports_warnings_without_failing(
    cli_runner: CliRunner,
    examples_dir: Path,
) -> None:
    result = cli_runner.invoke(
        app,
        ["compare", str(examples_dir / "train.csv"), str(examples_dir / "candidate.csv")],
    )

    assert result.exit_code == 0
    assert "Dataset Comparison" in result.stdout
    assert "Numeric drift signals: 2" in result.stdout
    assert "user_id" not in result.stdout


def test_compare_cli_fails_for_missing_columns(
    cli_runner: CliRunner,
    write_dataset,
    tmp_path: Path,
) -> None:
    train_path = write_dataset(
        pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4]}),
        "csv",
    )
    candidate_path = tmp_path / "candidate.csv"
    pd.DataFrame({"feature_a": [1, 2]}).to_csv(candidate_path, index=False)
    report_path = tmp_path / "compare.md"

    result = cli_runner.invoke(
        app,
        ["compare", str(train_path), str(candidate_path), "--report", str(report_path)],
    )

    assert result.exit_code == 1
    assert "Missing columns: 1" in result.stdout
    assert "Fix schema mismatches before training or rollout." in result.stdout
    assert report_path.exists()


def test_inference_payload_cli_fails_for_invalid_payload(
    cli_runner: CliRunner,
    examples_dir: Path,
) -> None:
    result = cli_runner.invoke(
        app,
        [
            "check-inference-payload",
            str(examples_dir / "inference_payload_invalid.json"),
            "--schema",
            str(examples_dir / "inference_schema.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Inference Payload Check" in result.stdout
    assert "Status: FAIL" in result.stdout


def test_cli_help_uses_preferred_report_flag_name(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["check-dataset", "--help"])

    assert result.exit_code == 0
    assert "--report" in result.stdout
    assert ".csv" in result.stdout
    assert ".parquet" in result.stdout


def test_cli_reports_missing_dataset_file_clearly(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["check-dataset", "examples/does-not-exist.csv"])

    assert result.exit_code == 1
    assert "What went wrong" in result.stderr
    assert "Could not find dataset file" in result.stderr
    assert "How to fix it" in result.stderr


def test_cli_reports_unsupported_dataset_format_clearly(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")

    result = cli_runner.invoke(app, ["check-dataset", str(dataset_path)])

    assert result.exit_code == 1
    assert "Unsupported dataset format" in result.stderr
    assert "Use a .csv or .parquet file" in result.stderr


def test_cli_reports_malformed_json_clearly(
    cli_runner: CliRunner,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "bad.json"
    payload_path.write_text('{"customer_id": 1,}', encoding="utf-8")

    result = cli_runner.invoke(
        app,
        [
            "check-inference-payload",
            str(payload_path),
            "--schema",
            str(examples_dir / "inference_schema.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Could not parse JSON" in result.stderr
    assert "Fix the JSON syntax and try again." in result.stderr


def test_cli_reports_invalid_schema_clearly(
    cli_runner: CliRunner,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"fields": {}}', encoding="utf-8")

    result = cli_runner.invoke(
        app,
        [
            "check-inference-payload",
            str(examples_dir / "inference_payload_valid.json"),
            "--schema",
            str(schema_path),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid schema definition" in result.stderr
    assert "top-level 'fields' mapping" in result.stderr
