from __future__ import annotations

from pathlib import Path

import pandas as pd

from orya_guard.checks.common import compute_null_ratios, is_constant_series
from orya_guard.models.results import DatasetCheckResult, DatasetProfile, Issue, IssueSeverity


def check_dataset(dataframe: pd.DataFrame, path: Path) -> DatasetCheckResult:
    """Inspect a dataset for common quality issues."""

    duplicate_row_count = int(dataframe.duplicated().sum())
    null_ratios = compute_null_ratios(dataframe)
    constant_columns = [
        column for column in dataframe.columns if is_constant_series(dataframe[column])
    ]

    issues: list[Issue] = []
    if duplicate_row_count > 0:
        issues.append(
            Issue(
                severity=IssueSeverity.WARNING,
                code="duplicate_rows",
                message=f"Found {duplicate_row_count} duplicate rows.",
            )
        )

    for column, ratio in sorted(null_ratios.items()):
        if ratio > 0:
            issues.append(
                Issue(
                    severity=IssueSeverity.WARNING,
                    code="null_ratio",
                    message=f"Column '{column}' has a {ratio:.1%} null ratio.",
                    column=column,
                )
            )

    for column in constant_columns:
        issues.append(
            Issue(
                severity=IssueSeverity.WARNING,
                code="constant_column",
                message=f"Column '{column}' is constant.",
                column=column,
            )
        )

    profile = DatasetProfile(
        path=str(path),
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        duplicate_row_count=duplicate_row_count,
        null_ratios=null_ratios,
        constant_columns=constant_columns,
    )
    return DatasetCheckResult(dataset=profile, issues=issues)
