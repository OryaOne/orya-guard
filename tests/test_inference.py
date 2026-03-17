from __future__ import annotations

from pathlib import Path

from orya_guard.checks.inference import validate_inference_payload
from orya_guard.io.data_loading import load_inference_schema, load_json_payload


def test_inference_validation_accepts_valid_payload(examples_dir: Path) -> None:
    schema = load_inference_schema(examples_dir / "inference_schema.json")
    payload = load_json_payload(examples_dir / "inference_payload_valid.json")

    result = validate_inference_payload(payload, schema, examples_dir / "valid.json")

    assert result.has_errors() is False
    assert result.missing_required_fields == []
    assert result.unexpected_fields == []
    assert result.type_errors == []


def test_inference_validation_reports_missing_unexpected_and_type_errors(
    examples_dir: Path,
) -> None:
    schema = load_inference_schema(examples_dir / "inference_schema.json")
    payload = load_json_payload(examples_dir / "inference_payload_invalid.json")

    result = validate_inference_payload(
        payload,
        schema,
        examples_dir / "inference_payload_invalid.json",
    )

    assert result.has_errors()
    assert result.missing_required_fields == ["age"]
    assert result.unexpected_fields == ["unexpected_feature"]
    assert {error.field for error in result.type_errors} == {"customer_id", "city"}


def test_inference_validation_rejects_non_object_payload(examples_dir: Path) -> None:
    schema = load_inference_schema(examples_dir / "inference_schema.json")

    result = validate_inference_payload(["not", "an", "object"], schema, Path("payload.json"))

    assert result.has_errors()
    assert result.missing_required_fields == []
    assert result.unexpected_fields == []
    assert result.type_errors == []
    assert any(issue.code == "invalid_payload_type" for issue in result.issues)
