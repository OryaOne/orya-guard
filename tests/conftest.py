from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def write_dataset(tmp_path: Path) -> Callable[[pd.DataFrame, Literal["csv", "parquet"]], Path]:
    def _write_dataset(dataframe: pd.DataFrame, file_format: Literal["csv", "parquet"]) -> Path:
        path = tmp_path / f"dataset.{file_format}"
        if file_format == "csv":
            dataframe.to_csv(path, index=False)
        else:
            dataframe.to_parquet(path, index=False)
        return path

    return _write_dataset
