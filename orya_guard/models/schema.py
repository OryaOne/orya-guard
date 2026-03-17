from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldType(str, Enum):
    """Supported field types for inference payload validation."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class SchemaField(BaseModel):
    """One field in the inference schema."""

    model_config = ConfigDict(extra="forbid")

    type: FieldType
    required: bool = True


class InferenceSchema(BaseModel):
    """Simple V1 payload schema model."""

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, SchemaField] = Field(...)

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, fields: dict[str, SchemaField]) -> dict[str, SchemaField]:
        if not fields:
            raise ValueError("Schema 'fields' must define at least one field.")
        return fields
