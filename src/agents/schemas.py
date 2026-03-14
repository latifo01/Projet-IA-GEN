"""Pydantic schemas for validating LLM proposals."""

from typing import Literal
from pydantic import BaseModel


# -- Transformation schemas --

TransformAction = Literal[
    "replace_sentinel",
    "cast_numeric",
    "impute_mean",
    "impute_median",
    "impute_mode",
    "drop_column",
    "drop_rows",
    "one_hot_encoding",
    "label_encoding",
    "ordinal_encoding",
    "standard_scaling",
    "minmax_scaling",
    "robust_scaling",
    "log_transform",
    "sqrt_transform",
    "frequency_encoding",
    "extract_datetime",
]


class TransformProposal(BaseModel):
    column: str
    action: TransformAction
    reason: str
    confidence: Literal["high", "medium", "low"] = "medium"
    source: Literal["rule", "llm"] = "llm"
    sentinel_values: list[float | int | str] = []


class TransformResponse(BaseModel):
    transformations: list[TransformProposal]


# -- Outlier schemas --

OutlierMethod = Literal["iqr", "zscore", "bimodal"]
OutlierAction = Literal["clip", "remove", "keep", "replace_with_nan"]


class OutlierProposal(BaseModel):
    column: str
    method: OutlierMethod
    action: OutlierAction
    reason: str
    confidence: Literal["high", "medium", "low"] = "medium"
    source: Literal["rule", "llm"] = "llm"


class OutlierResponse(BaseModel):
    outlier_actions: list[OutlierProposal]


# -- Domain context + semantic anomalies (merged, single LLM call) --

class SemanticAnomaly(BaseModel):
    column: str
    issue: str
    severity: Literal["high", "medium", "low"]
    type: Literal["proxy_for_missing", "impossible_value", "other"] = "other"
    sentinel_values: list[float | int | str] = []


class DomainAndAnomaliesResponse(BaseModel):
    domain: str
    description: str
    constraints: list[str]
    sensitive_columns: list[str]
    anomalies: list[SemanticAnomaly]


# -- Diagnosis schema (LLM call 2 in analyze) --

class DiagnosisResponse(BaseModel):
    summary: str
    issues: list[str]
    recommendations: list[str]


# -- Feature engineering proposal schema --

class FeatureProposal(BaseModel):
    name: str
    formula: str
    reason: str
    source_columns: list[str]


class FeatureEngineeringResponse(BaseModel):
    features: list[FeatureProposal]


# -- Leakage detection --

class LeakageAlert(BaseModel):
    column: str
    correlation: float
    reason: str


# -- Inter-feature correlation --

class CorrelationAlert(BaseModel):
    column_a: str
    column_b: str
    correlation: float
    method: str                # "pearson", "cramers_v"
    severity: Literal["high", "very_high"]
    suggested_actions: list[str]
