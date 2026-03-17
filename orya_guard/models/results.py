from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    """Severity levels used by validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Issue(BaseModel):
    """A human-friendly validation issue."""

    severity: IssueSeverity
    code: str
    message: str
    column: str | None = None


class CheckResultBase(BaseModel):
    """Shared result helpers for all validation commands."""

    issues: list[Issue] = Field(default_factory=list)

    def has_errors(self) -> bool:
        return any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        return any(issue.severity == IssueSeverity.WARNING for issue in self.issues)

    def status(self) -> str:
        if self.has_errors():
            return "FAIL"
        if self.issues:
            return "WARN"
        return "PASS"


class DatasetProfile(BaseModel):
    """Dataset-level statistics gathered during inspection."""

    path: str
    row_count: int
    column_count: int
    dtypes: dict[str, str]
    duplicate_row_count: int
    null_ratios: dict[str, float]
    constant_columns: list[str]


class DatasetCheckResult(CheckResultBase):
    """Result of a single dataset inspection run."""

    dataset: DatasetProfile


class DtypeMismatch(BaseModel):
    """A shared column with conflicting dtypes."""

    column: str
    train_dtype: str
    candidate_dtype: str


class NullRatioChange(BaseModel):
    """A significant null ratio change between datasets."""

    column: str
    train_ratio: float
    candidate_ratio: float
    delta: float


class UnseenCategoricalValue(BaseModel):
    """Unseen candidate categories for a column."""

    column: str
    unseen_values: list[str]
    count: int


class NumericDrift(BaseModel):
    """A simple numeric drift signal for a column."""

    column: str
    train_mean: float
    candidate_mean: float
    train_std: float
    candidate_std: float
    reason: str


class CompareResult(CheckResultBase):
    """Result of comparing a train dataset to a candidate dataset."""

    train_path: str
    candidate_path: str
    missing_columns: list[str] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    dtype_mismatches: list[DtypeMismatch] = Field(default_factory=list)
    null_ratio_changes: list[NullRatioChange] = Field(default_factory=list)
    unseen_categorical_values: list[UnseenCategoricalValue] = Field(default_factory=list)
    numeric_drifts: list[NumericDrift] = Field(default_factory=list)


class TypeErrorDetail(BaseModel):
    """A field-level inference payload type mismatch."""

    field: str
    expected: str
    actual: str


class InferenceValidationResult(CheckResultBase):
    """Result of validating one inference payload."""

    payload_path: str
    missing_required_fields: list[str] = Field(default_factory=list)
    unexpected_fields: list[str] = Field(default_factory=list)
    type_errors: list[TypeErrorDetail] = Field(default_factory=list)
