from __future__ import annotations

from pathlib import Path
from typing import Any

from orya_guard.models.results import (
    InferenceValidationResult,
    Issue,
    IssueSeverity,
    TypeErrorDetail,
)
from orya_guard.models.schema import FieldType, InferenceSchema


def validate_inference_payload(
    payload: Any,
    schema: InferenceSchema,
    payload_path: Path,
) -> InferenceValidationResult:
    """Validate one inference payload against a simple schema."""

    issues: list[Issue] = []
    missing_required_fields: list[str] = []
    unexpected_fields: list[str] = []
    type_errors: list[TypeErrorDetail] = []

    if not isinstance(payload, dict):
        issues.append(
            Issue(
                severity=IssueSeverity.ERROR,
                code="invalid_payload_type",
                message="Inference payload must be a JSON object.",
            )
        )
        return InferenceValidationResult(
            payload_path=str(payload_path),
            issues=issues,
            missing_required_fields=missing_required_fields,
            unexpected_fields=unexpected_fields,
            type_errors=type_errors,
        )

    for field_name, field_schema in schema.fields.items():
        if field_schema.required and field_name not in payload:
            missing_required_fields.append(field_name)
            issues.append(
                Issue(
                    severity=IssueSeverity.ERROR,
                    code="missing_required_field",
                    message=f"Missing required field '{field_name}'.",
                    column=field_name,
                )
            )

    for field_name in sorted(payload):
        if field_name not in schema.fields:
            unexpected_fields.append(field_name)
            issues.append(
                Issue(
                    severity=IssueSeverity.ERROR,
                    code="unexpected_field",
                    message=f"Unexpected field '{field_name}' was provided.",
                    column=field_name,
                )
            )
            continue

        expected_type = schema.fields[field_name].type
        value = payload[field_name]
        if not _matches_type(value, expected_type):
            detail = TypeErrorDetail(
                field=field_name,
                expected=expected_type.value,
                actual=_describe_value_type(value),
            )
            type_errors.append(detail)
            issues.append(
                Issue(
                    severity=IssueSeverity.ERROR,
                    code="type_mismatch",
                    message=(
                        f"Field '{field_name}' should be '{expected_type.value}' "
                        f"but received '{detail.actual}'."
                    ),
                    column=field_name,
                )
            )

    return InferenceValidationResult(
        payload_path=str(payload_path),
        missing_required_fields=missing_required_fields,
        unexpected_fields=unexpected_fields,
        type_errors=type_errors,
        issues=issues,
    )


def _matches_type(value: Any, expected_type: FieldType) -> bool:
    if expected_type == FieldType.STRING:
        return isinstance(value, str)
    if expected_type == FieldType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == FieldType.NUMBER:
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected_type == FieldType.BOOLEAN:
        return isinstance(value, bool)
    if expected_type == FieldType.OBJECT:
        return isinstance(value, dict)
    if expected_type == FieldType.ARRAY:
        return isinstance(value, list)
    return False


def _describe_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__
