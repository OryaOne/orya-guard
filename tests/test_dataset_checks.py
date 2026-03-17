from __future__ import annotations

from pathlib import Path

import pandas as pd

from orya_guard.checks.dataset import check_dataset
from orya_guard.io.data_loading import load_dataframe


def test_check_dataset_detects_duplicate_rows_null_ratios_and_constant_columns(
    examples_dir: Path,
) -> None:
    dataframe = load_dataframe(examples_dir / "dataset_with_issues.csv")

    result = check_dataset(dataframe, examples_dir / "dataset_with_issues.csv")

    assert result.dataset.row_count == 4
    assert result.dataset.column_count == 4
    assert result.dataset.duplicate_row_count == 1
    assert result.dataset.null_ratios["score"] == 0.5
    assert "status" in result.dataset.constant_columns
    assert "constant_feature" in result.dataset.constant_columns
    assert any(issue.code == "duplicate_rows" for issue in result.issues)


def test_check_dataset_returns_no_issues_for_clean_dataset() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "score": [0.2, 0.4, 0.8],
            "city": ["athens", "patras", "larisa"],
        }
    )

    result = check_dataset(dataframe, Path("clean.csv"))

    assert result.dataset.duplicate_row_count == 0
    assert result.dataset.constant_columns == []
    assert result.dataset.null_ratios == {"customer_id": 0.0, "score": 0.0, "city": 0.0}
    assert result.issues == []


def test_load_dataframe_supports_parquet(examples_dir: Path, tmp_path: Path) -> None:
    source = load_dataframe(examples_dir / "train.csv")
    parquet_path = tmp_path / "train.parquet"
    source.to_parquet(parquet_path, index=False)

    loaded = load_dataframe(parquet_path)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == source.shape
