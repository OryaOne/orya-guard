from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from orya_guard.errors import OryaGuardError
from orya_guard.models.schema import InferenceSchema


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a supported tabular dataset from disk."""

    resolved_path = path.expanduser()
    _validate_file_path(resolved_path, "dataset")

    suffix = resolved_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(resolved_path)
        if suffix == ".parquet":
            return pd.read_parquet(resolved_path)
    except Exception as error:  # pragma: no cover - pandas error details vary
        raise OryaGuardError(
            message=f"Could not read dataset file '{resolved_path}'.",
            next_step="Check that the file is a valid CSV or parquet dataset and try again.",
        ) from error

    raise OryaGuardError(
        message=f"Unsupported dataset format for '{resolved_path.name}'.",
        next_step="Use a .csv or .parquet file, then run the command again.",
    )


def load_json_payload(path: Path) -> Any:
    """Load a JSON file from disk."""

    resolved_path = path.expanduser()
    _validate_file_path(resolved_path, "JSON")

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except JSONDecodeError as error:
        raise OryaGuardError(
            message=(
                f"Could not parse JSON in '{resolved_path}' "
                f"(line {error.lineno}, column {error.colno})."
            ),
            next_step="Fix the JSON syntax and try again.",
        ) from error


def load_inference_schema(path: Path) -> InferenceSchema:
    """Load and validate an inference schema file."""

    schema_data = load_json_payload(path)
    try:
        return InferenceSchema.model_validate(schema_data)
    except ValidationError as error:
        raise OryaGuardError(
            message=(
                f"Invalid schema definition in '{path.expanduser()}': "
                f"{_summarize_schema_error(error)}"
            ),
            next_step=(
                "Use a JSON object with a top-level 'fields' mapping. "
                "Each field must declare a supported 'type' and may set 'required'."
            ),
        ) from error


def _validate_file_path(path: Path, label: str) -> None:
    if not path.exists():
        raise OryaGuardError(
            message=f"Could not find {label} file '{path}'.",
            next_step="Check the path and try again.",
        )
    if not path.is_file():
        raise OryaGuardError(
            message=f"Expected a file at '{path}', but found something else.",
            next_step="Pass a file path instead of a directory and try again.",
        )


def _summarize_schema_error(error: ValidationError) -> str:
    first_error = error.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    return f"{location}: {first_error['msg']}"
