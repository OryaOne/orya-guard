from __future__ import annotations

from pathlib import Path

import pandas as pd

from orya_guard.checks.common import (
    are_compatible_dtypes,
    compute_null_ratios,
    is_categorical_series,
    is_identifier_like_column,
)
from orya_guard.models.results import (
    CompareResult,
    DtypeMismatch,
    Issue,
    IssueSeverity,
    NullRatioChange,
    NumericDrift,
    UnseenCategoricalValue,
)

NULL_SPIKE_THRESHOLD = 0.10
MEAN_SHIFT_STD_THRESHOLD = 1.0
MEDIAN_SHIFT_STD_THRESHOLD = 1.0
STD_RATIO_UPPER_THRESHOLD = 2.0
STD_RATIO_LOWER_THRESHOLD = 0.5


def compare_datasets(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    train_path: Path,
    candidate_path: Path,
) -> CompareResult:
    """Compare a candidate dataset against a training reference dataset."""

    missing_columns, extra_columns, common_columns = _resolve_column_sets(train_df, candidate_df)
    null_ratio_changes: list[NullRatioChange] = []
    unseen_categorical_values: list[UnseenCategoricalValue] = []
    dtype_mismatches: list[DtypeMismatch] = []
    numeric_drifts: list[NumericDrift] = []

    issues = _collect_schema_issues(missing_columns, extra_columns)
    train_null_ratios = compute_null_ratios(train_df)
    candidate_null_ratios = compute_null_ratios(candidate_df)

    for column in common_columns:
        train_series = train_df[column]
        candidate_series = candidate_df[column]
        dtype_mismatch = _build_dtype_mismatch(column, train_series, candidate_series)
        if dtype_mismatch is not None:
            dtype_mismatches.append(dtype_mismatch)
            issues.append(_dtype_mismatch_issue(dtype_mismatch))
            continue

        null_ratio_change = _build_null_ratio_change(
            column,
            train_null_ratios[column],
            candidate_null_ratios[column],
        )
        if null_ratio_change is not None:
            null_ratio_changes.append(null_ratio_change)
            issues.append(_null_ratio_change_issue(null_ratio_change))

        unseen_values = _build_unseen_categorical_values(column, train_series, candidate_series)
        if unseen_values is not None:
            unseen_categorical_values.append(unseen_values)
            issues.append(_unseen_categorical_values_issue(unseen_values))

        if not _should_check_numeric_drift(column, train_series, candidate_series):
            continue

        drift = _detect_numeric_drift(train_series, candidate_series, column)
        if drift is not None:
            numeric_drifts.append(drift)
            issues.append(_numeric_drift_issue(drift))

    return CompareResult(
        train_path=str(train_path),
        candidate_path=str(candidate_path),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        dtype_mismatches=dtype_mismatches,
        null_ratio_changes=null_ratio_changes,
        unseen_categorical_values=unseen_categorical_values,
        numeric_drifts=numeric_drifts,
        issues=issues,
    )


def _resolve_column_sets(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    train_columns = set(train_df.columns)
    candidate_columns = set(candidate_df.columns)
    missing_columns = sorted(train_columns - candidate_columns)
    extra_columns = sorted(candidate_columns - train_columns)
    common_columns = sorted(train_columns & candidate_columns)
    return missing_columns, extra_columns, common_columns


def _collect_schema_issues(missing_columns: list[str], extra_columns: list[str]) -> list[Issue]:
    issues: list[Issue] = []
    if missing_columns:
        issues.append(
            Issue(
                severity=IssueSeverity.ERROR,
                code="missing_columns",
                message=f"Candidate dataset is missing columns: {', '.join(missing_columns)}.",
            )
        )
    if extra_columns:
        issues.append(
            Issue(
                severity=IssueSeverity.WARNING,
                code="extra_columns",
                message=f"Candidate dataset has extra columns: {', '.join(extra_columns)}.",
            )
        )
    return issues


def _build_dtype_mismatch(
    column: str,
    train_series: pd.Series,
    candidate_series: pd.Series,
) -> DtypeMismatch | None:
    if are_compatible_dtypes(train_series, candidate_series):
        return None
    return DtypeMismatch(
        column=column,
        train_dtype=str(train_series.dtype),
        candidate_dtype=str(candidate_series.dtype),
    )


def _dtype_mismatch_issue(mismatch: DtypeMismatch) -> Issue:
    return Issue(
        severity=IssueSeverity.ERROR,
        code="dtype_mismatch",
        message=(
            f"Column '{mismatch.column}' has dtype '{mismatch.candidate_dtype}' in candidate "
            f"data but '{mismatch.train_dtype}' in training data."
        ),
        column=mismatch.column,
    )


def _build_null_ratio_change(
    column: str,
    train_ratio: float,
    candidate_ratio: float,
) -> NullRatioChange | None:
    null_delta = abs(candidate_ratio - train_ratio)
    if null_delta < NULL_SPIKE_THRESHOLD:
        return None
    return NullRatioChange(
        column=column,
        train_ratio=round(train_ratio, 4),
        candidate_ratio=round(candidate_ratio, 4),
        delta=round(null_delta, 4),
    )


def _null_ratio_change_issue(change: NullRatioChange) -> Issue:
    return Issue(
        severity=IssueSeverity.WARNING,
        code="null_spike",
        message=(
            f"Column '{change.column}' null ratio changed from "
            f"{change.train_ratio:.1%} to {change.candidate_ratio:.1%}."
        ),
        column=change.column,
    )


def _build_unseen_categorical_values(
    column: str,
    train_series: pd.Series,
    candidate_series: pd.Series,
) -> UnseenCategoricalValue | None:
    if not is_categorical_series(train_series):
        return None

    unseen = sorted(
        {str(value) for value in candidate_series.dropna().unique()}
        - {str(value) for value in train_series.dropna().unique()}
    )
    if not unseen:
        return None

    return UnseenCategoricalValue(
        column=column,
        unseen_values=unseen[:10],
        count=len(unseen),
    )


def _unseen_categorical_values_issue(unseen_values: UnseenCategoricalValue) -> Issue:
    preview = ", ".join(unseen_values.unseen_values[:5])
    return Issue(
        severity=IssueSeverity.WARNING,
        code="unseen_categorical_values",
        message=(f"Column '{unseen_values.column}' contains unseen candidate values: {preview}."),
        column=unseen_values.column,
    )


def _should_check_numeric_drift(
    column: str,
    train_series: pd.Series,
    candidate_series: pd.Series,
) -> bool:
    if not pd.api.types.is_numeric_dtype(train_series):
        return False
    if not pd.api.types.is_numeric_dtype(candidate_series):
        return False
    if is_identifier_like_column(column, train_series) or is_identifier_like_column(
        column, candidate_series
    ):
        return False
    return True


def _numeric_drift_issue(drift: NumericDrift) -> Issue:
    return Issue(
        severity=IssueSeverity.WARNING,
        code="numeric_drift",
        message=f"Column '{drift.column}' shows suspicious numeric drift: {drift.reason}.",
        column=drift.column,
    )


def _detect_numeric_drift(
    train_series: pd.Series,
    candidate_series: pd.Series,
    column: str,
) -> NumericDrift | None:
    train_clean = train_series.dropna()
    candidate_clean = candidate_series.dropna()
    if train_clean.empty or candidate_clean.empty:
        return None

    train_mean = float(train_clean.mean())
    candidate_mean = float(candidate_clean.mean())
    train_std = float(train_clean.std(ddof=0))
    candidate_std = float(candidate_clean.std(ddof=0))
    train_median = float(train_clean.median())
    candidate_median = float(candidate_clean.median())

    reasons: list[str] = []
    if train_std == 0:
        if any(float(value) != train_mean for value in candidate_clean.tolist()):
            reasons.append("training data is constant but candidate data varies")
    else:
        mean_shift = abs(candidate_mean - train_mean) / max(train_std, 1e-9)
        median_shift = abs(candidate_median - train_median) / max(train_std, 1e-9)
        std_ratio = candidate_std / max(train_std, 1e-9)

        if mean_shift >= MEAN_SHIFT_STD_THRESHOLD:
            reasons.append("mean shifted by at least one training standard deviation")
        if median_shift >= MEDIAN_SHIFT_STD_THRESHOLD:
            reasons.append("median shifted by at least one training standard deviation")
        if std_ratio >= STD_RATIO_UPPER_THRESHOLD or std_ratio <= STD_RATIO_LOWER_THRESHOLD:
            reasons.append("spread changed materially versus training data")

    if not reasons:
        return None

    return NumericDrift(
        column=column,
        train_mean=round(train_mean, 4),
        candidate_mean=round(candidate_mean, 4),
        train_std=round(train_std, 4),
        candidate_std=round(candidate_std, 4),
        reason="; ".join(reasons),
    )
