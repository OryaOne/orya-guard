from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orya_guard.errors import OryaGuardError
from orya_guard.io.data_loading import load_dataframe, load_inference_schema, load_json_payload


def test_load_dataframe_reads_csv_and_parquet(
    write_dataset,
) -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "score": [0.1, 0.2],
        }
    )
    csv_path = write_dataset(dataframe, "csv")
    parquet_path = write_dataset(dataframe, "parquet")

    csv_result = load_dataframe(csv_path)
    parquet_result = load_dataframe(parquet_path)

    assert csv_result.to_dict(orient="records") == parquet_result.to_dict(orient="records")


def test_load_dataframe_rejects_missing_file() -> None:
    with pytest.raises(OryaGuardError, match="Could not find dataset file"):
        load_dataframe(Path("missing.csv"))


def test_load_dataframe_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(OryaGuardError, match="Unsupported dataset format"):
        load_dataframe(path)


def test_load_dataframe_reports_corrupt_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(OryaGuardError, match="Could not read dataset file"):
        load_dataframe(path)


def test_load_json_payload_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"customer_id": 1,}', encoding="utf-8")

    with pytest.raises(OryaGuardError, match="Could not parse JSON"):
        load_json_payload(path)


def test_load_inference_schema_rejects_empty_fields_mapping(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_text('{"fields": {}}', encoding="utf-8")

    with pytest.raises(OryaGuardError, match="Invalid schema definition"):
        load_inference_schema(path)


def test_load_inference_schema_rejects_unsupported_field_type(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_text(
        '{"fields": {"customer_id": {"type": "uuid", "required": true}}}',
        encoding="utf-8",
    )

    with pytest.raises(OryaGuardError, match="Invalid schema definition"):
        load_inference_schema(path)
