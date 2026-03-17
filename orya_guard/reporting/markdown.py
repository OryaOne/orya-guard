from __future__ import annotations

from orya_guard.models.results import CompareResult, DatasetCheckResult


def render_dataset_markdown(result: DatasetCheckResult) -> str:
    """Render a Markdown report for dataset inspection."""

    dtype_lines = "\n".join(
        f"| `{column}` | `{dtype}` |" for column, dtype in sorted(result.dataset.dtypes.items())
    )
    null_ratio_lines = "\n".join(
        f"| `{column}` | {ratio:.1%} |"
        for column, ratio in sorted(result.dataset.null_ratios.items())
    )
    constant_columns = (
        "\n".join(f"- `{column}`" for column in result.dataset.constant_columns) or "- None"
    )
    findings = "\n".join(f"- {issue.message}" for issue in result.issues) or "- None detected."

    return "\n".join(
        [
            "# Dataset Check Report",
            "",
            "## Summary",
            "",
            f"- Path: `{result.dataset.path}`",
            f"- Rows: `{result.dataset.row_count}`",
            f"- Columns: `{result.dataset.column_count}`",
            f"- Duplicate rows: `{result.dataset.duplicate_row_count}`",
            "",
            "## Column dtypes",
            "",
            "| Column | Dtype |",
            "| --- | --- |",
            dtype_lines,
            "",
            "## Null ratios",
            "",
            "| Column | Null ratio |",
            "| --- | --- |",
            null_ratio_lines,
            "",
            "## Constant columns",
            "",
            constant_columns,
            "",
            "## Findings",
            "",
            findings,
            "",
        ]
    )


def render_compare_markdown(result: CompareResult) -> str:
    """Render a Markdown report for dataset comparison."""

    findings = "\n".join(f"- {issue.message}" for issue in result.issues) or "- None detected."
    missing_columns = "\n".join(f"- `{column}`" for column in result.missing_columns) or "- None"
    extra_columns = "\n".join(f"- `{column}`" for column in result.extra_columns) or "- None"
    dtype_mismatches = (
        "\n".join(
            f"| `{mismatch.column}` | `{mismatch.train_dtype}` | `{mismatch.candidate_dtype}` |"
            for mismatch in result.dtype_mismatches
        )
        or "| None | - | - |"
    )
    null_changes = (
        "\n".join(
            (
                f"| `{change.column}` | {change.train_ratio:.1%} | "
                f"{change.candidate_ratio:.1%} | {change.delta:.1%} |"
            )
            for change in result.null_ratio_changes
        )
        or "| None | - | - | - |"
    )
    unseen_values = (
        "\n".join(
            (
                f"| `{item.column}` | {item.count} | "
                f"{', '.join(f'`{value}`' for value in item.unseen_values)} |"
            )
            for item in result.unseen_categorical_values
        )
        or "| None | - | - |"
    )
    drifts = (
        "\n".join(
            (
                f"| `{drift.column}` | `{drift.train_mean}` | "
                f"`{drift.candidate_mean}` | `{drift.reason}` |"
            )
            for drift in result.numeric_drifts
        )
        or "| None | - | - | - |"
    )

    return "\n".join(
        [
            "# Dataset Comparison Report",
            "",
            "## Summary",
            "",
            f"- Train: `{result.train_path}`",
            f"- Candidate: `{result.candidate_path}`",
            "",
            "## Missing columns",
            "",
            missing_columns,
            "",
            "## Extra columns",
            "",
            extra_columns,
            "",
            "## Dtype mismatches",
            "",
            "| Column | Train dtype | Candidate dtype |",
            "| --- | --- | --- |",
            dtype_mismatches,
            "",
            "## Null ratio changes",
            "",
            "| Column | Train | Candidate | Delta |",
            "| --- | --- | --- | --- |",
            null_changes,
            "",
            "## Unseen categorical values",
            "",
            "| Column | Count | Example values |",
            "| --- | --- | --- |",
            unseen_values,
            "",
            "## Numeric drift",
            "",
            "| Column | Train mean | Candidate mean | Reason |",
            "| --- | --- | --- | --- |",
            drifts,
            "",
            "## Findings",
            "",
            findings,
            "",
        ]
    )
