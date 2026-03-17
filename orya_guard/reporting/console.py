from __future__ import annotations

from orya_guard.models.results import (
    CompareResult,
    DatasetCheckResult,
    InferenceValidationResult,
    Issue,
)


def render_dataset_summary(result: DatasetCheckResult) -> str:
    """Render a CLI summary for dataset inspection."""

    columns_with_nulls = [
        column for column, ratio in result.dataset.null_ratios.items() if ratio > 0
    ]
    lines = [
        "Dataset Check",
        f"Status: {result.status()}",
    ]
    lines.extend(
        _render_section(
            "Summary",
            [
                f"Dataset: {result.dataset.path}",
                f"Rows: {result.dataset.row_count}",
                f"Columns: {result.dataset.column_count}",
                f"Column dtypes: {_format_mapping(result.dataset.dtypes)}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(
        _render_section(
            "Key findings",
            [
                f"Duplicate rows: {result.dataset.duplicate_row_count}",
                f"Columns with nulls: {len(columns_with_nulls)}",
                f"Constant columns: {len(result.dataset.constant_columns)}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(_render_findings_section(result.issues))
    lines.extend(
        _render_section(
            "Next step",
            [_next_step_for_dataset(result)],
            include_leading_blank=True,
        )
    )
    return "\n".join(lines)


def render_compare_summary(result: CompareResult) -> str:
    """Render a CLI summary for dataset comparison."""

    lines = [
        "Dataset Comparison",
        f"Status: {result.status()}",
    ]
    lines.extend(
        _render_section(
            "Summary",
            [
                f"Training dataset: {result.train_path}",
                f"Candidate dataset: {result.candidate_path}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(
        _render_section(
            "Key findings",
            [
                f"Missing columns: {len(result.missing_columns)}",
                f"Extra columns: {len(result.extra_columns)}",
                f"Incompatible dtypes: {len(result.dtype_mismatches)}",
                f"Null ratio changes: {len(result.null_ratio_changes)}",
                f"Unseen categorical values: {len(result.unseen_categorical_values)}",
                f"Numeric drift signals: {len(result.numeric_drifts)}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(_render_findings_section(result.issues))
    lines.extend(
        _render_section(
            "Next step",
            [_next_step_for_compare(result)],
            include_leading_blank=True,
        )
    )
    return "\n".join(lines)


def render_inference_summary(
    result: InferenceValidationResult,
    schema_path: str,
) -> str:
    """Render a CLI summary for inference payload validation."""

    lines = [
        "Inference Payload Check",
        f"Status: {result.status()}",
    ]
    lines.extend(
        _render_section(
            "Summary",
            [
                f"Payload: {result.payload_path}",
                f"Schema: {schema_path}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(
        _render_section(
            "Key findings",
            [
                f"Missing required fields: {len(result.missing_required_fields)}",
                f"Unexpected fields: {len(result.unexpected_fields)}",
                f"Type mismatches: {len(result.type_errors)}",
            ],
            include_leading_blank=True,
        )
    )
    lines.extend(_render_findings_section(result.issues))
    lines.extend(
        _render_section("Next step", [_next_step_for_inference(result)], include_leading_blank=True)
    )
    return "\n".join(lines)


def _render_issue_lines(issues: list[Issue]) -> list[str]:
    return [f"- {issue.message}" for issue in issues]


def _render_section(
    title: str,
    items: list[str],
    *,
    include_leading_blank: bool,
) -> list[str]:
    lines: list[str] = []
    if include_leading_blank:
        lines.append("")
    lines.extend([title, "-" * len(title)])
    lines.extend(f"- {item}" for item in items)
    return lines


def _render_findings_section(issues: list[Issue]) -> list[str]:
    lines = ["", "Detailed findings", "-----------------"]
    if issues:
        lines.extend(_render_issue_lines(issues))
    else:
        lines.append("- No issues detected.")
    return lines


def _format_mapping(values: dict[str, str], limit: int = 6) -> str:
    items = [f"{key}={value}" for key, value in sorted(values.items())]
    return _format_items(items, limit=limit)


def _format_items(values: list[str], limit: int = 6) -> str:
    if not values:
        return "none"
    if len(values) <= limit:
        return ", ".join(values)
    visible = ", ".join(values[:limit])
    remaining = len(values) - limit
    return f"{visible}, +{remaining} more"


def _next_step_for_dataset(result: DatasetCheckResult) -> str:
    if result.issues:
        return "Review duplicate rows and flagged columns before using this dataset."
    return "No action is required."


def _next_step_for_compare(result: CompareResult) -> str:
    if result.has_errors():
        return "Fix schema mismatches before training or rollout."
    if result.issues:
        return "Review the flagged columns before promoting the candidate dataset."
    return "The candidate dataset looks compatible with the training reference."


def _next_step_for_inference(result: InferenceValidationResult) -> str:
    if result.has_errors():
        return "Fix the payload or schema so every required field and type matches."
    return "The payload matches the schema and is ready for use."
