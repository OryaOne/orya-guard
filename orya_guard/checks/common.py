from __future__ import annotations

import re

import pandas as pd

IDENTIFIER_NAME_PATTERN = re.compile(r"(^id$|_id$|^index$|_key$)", re.IGNORECASE)


def compute_null_ratios(dataframe: pd.DataFrame) -> dict[str, float]:
    """Return rounded null ratios for each column in a dataframe."""

    return {
        column: round(float(dataframe[column].isna().mean()), 4) for column in dataframe.columns
    }


def is_constant_series(series: pd.Series) -> bool:
    """Return True when a series contains only one distinct value."""

    return bool(series.nunique(dropna=False) <= 1)


def is_categorical_series(series: pd.Series) -> bool:
    """Return True for series treated as categorical in V1 checks."""

    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def are_compatible_dtypes(train_series: pd.Series, candidate_series: pd.Series) -> bool:
    """Return True when two series are compatible for schema comparison."""

    if pd.api.types.is_bool_dtype(train_series) and pd.api.types.is_bool_dtype(candidate_series):
        return True
    if pd.api.types.is_numeric_dtype(train_series) and pd.api.types.is_numeric_dtype(
        candidate_series
    ):
        return True
    if is_categorical_series(train_series) and is_categorical_series(candidate_series):
        return True
    if pd.api.types.is_datetime64_any_dtype(train_series) and pd.api.types.is_datetime64_any_dtype(
        candidate_series
    ):
        return True
    return str(train_series.dtype) == str(candidate_series.dtype)


def is_identifier_like_column(column_name: str, series: pd.Series) -> bool:
    """Return True when a column likely represents row identity rather than a feature."""

    del series
    return bool(IDENTIFIER_NAME_PATTERN.search(column_name))
