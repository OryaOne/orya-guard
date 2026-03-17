from __future__ import annotations

from pathlib import Path

import pandas as pd

from orya_guard.checks.comparison import compare_datasets
from orya_guard.io.data_loading import load_dataframe


def test_compare_detects_null_ratio_changes_unseen_categories_and_numeric_drift(
    examples_dir: Path,
) -> None:
    train_df = load_dataframe(examples_dir / "train.csv")
    candidate_df = load_dataframe(examples_dir / "candidate.csv")

    result = compare_datasets(
        train_df,
        candidate_df,
        examples_dir / "train.csv",
        examples_dir / "candidate.csv",
    )

    assert result.missing_columns == []
    assert result.extra_columns == []
    assert result.dtype_mismatches == []
    assert any(change.column == "income" for change in result.null_ratio_changes)
    assert any(
        item.column == "city" and "volos" in item.unseen_values
        for item in result.unseen_categorical_values
    )
    assert any(drift.column == "income" for drift in result.numeric_drifts)
    assert all(drift.column != "user_id" for drift in result.numeric_drifts)
    assert not result.has_errors()


def test_compare_detects_missing_and_extra_columns() -> None:
    train_df = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_c": [9, 10],
        }
    )

    result = compare_datasets(train_df, candidate_df, Path("train.csv"), Path("candidate.csv"))

    assert result.missing_columns == ["feature_b"]
    assert result.extra_columns == ["feature_c"]
    assert result.has_errors()
    assert any(issue.code == "missing_columns" for issue in result.issues)
    assert any(issue.code == "extra_columns" for issue in result.issues)


def test_compare_detects_incompatible_dtypes_for_shared_columns() -> None:
    train_df = pd.DataFrame({"age": [21, 35], "city": ["athens", "patras"]})
    candidate_df = pd.DataFrame({"age": ["21", "35"], "city": ["athens", "patras"]})

    result = compare_datasets(train_df, candidate_df, Path("train.csv"), Path("candidate.csv"))

    assert len(result.dtype_mismatches) == 1
    assert result.dtype_mismatches[0].column == "age"
    assert result.has_errors()
    assert any(issue.code == "dtype_mismatch" for issue in result.issues)


def test_compare_treats_int_and_float_columns_as_compatible_numeric_types() -> None:
    train_df = pd.DataFrame({"score": [1, 2, 3]})
    candidate_df = pd.DataFrame({"score": [1.0, 2.0, None]})

    result = compare_datasets(train_df, candidate_df, Path("train.csv"), Path("candidate.csv"))

    assert result.dtype_mismatches == []
    assert any(change.column == "score" for change in result.null_ratio_changes)


def test_compare_returns_clean_result_for_matching_datasets() -> None:
    train_df = pd.DataFrame({"age": [21, 28, 35], "city": ["athens", "patras", "larisa"]})
    candidate_df = pd.DataFrame({"age": [22, 29, 36], "city": ["athens", "patras", "larisa"]})

    result = compare_datasets(train_df, candidate_df, Path("train.csv"), Path("candidate.csv"))

    assert result.missing_columns == []
    assert result.extra_columns == []
    assert result.dtype_mismatches == []
    assert result.null_ratio_changes == []
    assert result.unseen_categorical_values == []
    assert result.numeric_drifts == []
    assert result.issues == []


def test_compare_skips_numeric_drift_for_identifier_like_columns() -> None:
    train_df = pd.DataFrame({"customer_id": [1, 2, 3], "score": [0.2, 0.3, 0.4]})
    candidate_df = pd.DataFrame({"customer_id": [100, 101, 102], "score": [0.2, 0.3, 0.4]})

    result = compare_datasets(train_df, candidate_df, Path("train.csv"), Path("candidate.csv"))

    assert result.numeric_drifts == []
