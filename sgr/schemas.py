from __future__ import annotations

from typing import List, Dict, Any, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class AnalysisEntity(BaseModel):
    name: str
    type: Literal["table", "column", "metric", "date", "category", "number"]
    value: str


class AnalysisResult(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str
    intent: str
    entities: List[AnalysisEntity] = []
    filters: Dict[str, Any] = {}
    user_constraints: List[str] = []
    ambiguity_flags: List[str] = []


class StrategyPlan(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str
    pattern: Literal[
        "lookup",
        "aggregation",
        "timeseries",
        "join",
        "topk",
        "explain_only",
    ]
    rationale: str
    risks: List[str] = []
    chosen_model: str


class GenerationResult(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str
    sql_parametrized: str
    params: Dict[str, Any] = {}
    rationale: str
    expected_columns: List[str]
    expected_rowcount_hint: Optional[int] = None


class SafetyCheck(BaseModel):
    name: str
    passed: bool
    details: Optional[str] = None


class ValidationReport(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str
    passed: bool
    checks: List[SafetyCheck]
    execution_ms: Optional[int] = None
    sample_rows_preview: Optional[List[Dict[str, Any]]] = None
    truncation: bool = False
    error: Optional[str] = None


class FinalAnswer(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str
    answer_text: str
    sql_parametrized: Optional[str] = None
    params: Dict[str, Any] = {}
    columns: Optional[List[str]] = None
