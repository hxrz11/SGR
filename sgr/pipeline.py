from __future__ import annotations

from typing import Dict, Any

from .schemas import AnalysisResult, StrategyPlan, GenerationResult, ValidationReport, FinalAnswer
from .sql_guard import SQLGuard, GuardConfig
from .tracing import trace

# Заглушки стадий (LLM-вызовы вставьте внутри)


@trace("analysis")
def run_analysis(trace_id: str, user_query: str) -> Dict[str, Any]:
    # В реальности — вызов модели с промптом ANALYSIS_SYSTEM и валидация JSON
    return AnalysisResult(
        trace_id=trace_id,
        intent="user_question",
        entities=[],
    ).model_dump()


@trace("strategy")
def run_strategy(trace_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return StrategyPlan(
        trace_id=trace_id,
        pattern="lookup",
        rationale="simple lookup",
        chosen_model="gpt-4o-mini",
    ).model_dump()


@trace("generation")
def run_generation(trace_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
    # Здесь вы получаете от LLM параметризованный SQL + params
    return GenerationResult(
        trace_id=trace_id,
        sql_parametrized="SELECT id, name FROM public.users WHERE created_at >= %(dfrom)s",
        params={"dfrom": "2024-01-01"},
        rationale="filter by date",
        expected_columns=["id", "name"],
    ).model_dump()


@trace("validation")
def run_validation(trace_id: str, generation: Dict[str, Any], guard: SQLGuard) -> Dict[str, Any]:
    sql = generation["sql_parametrized"]
    guard.validate(sql)
    # Здесь можно добавить EXPLAIN, таймауты и пост-проверку схемы
    return ValidationReport(
        trace_id=trace_id,
        passed=True,
        checks=[{"name": "ast", "passed": True}],
    ).model_dump()


@trace("final")
def build_final_answer(trace_id: str, generation: Dict[str, Any]) -> Dict[str, Any]:
    return FinalAnswer(
        trace_id=trace_id,
        answer_text="Готово. Запрос подготовлен и проверен",
        sql_parametrized=generation["sql_parametrized"],
        params=generation["params"],
        columns=generation.get("expected_columns"),
    ).model_dump()


def run_pipeline(trace_id: str, user_query: str) -> Dict[str, Any]:
    analysis = run_analysis(trace_id=trace_id, user_query=user_query)
    strategy = run_strategy(trace_id=trace_id, analysis=analysis)
    generation = run_generation(trace_id=trace_id, strategy=strategy)

    guard = SQLGuard(
        GuardConfig(
            allowed_schemas={"public"},
            allowed_tables={"users"},
            allowed_columns={"id", "name", "created_at"},
            allowed_functions={"COUNT", "SUM", "AVG", "MIN", "MAX"},
            deny_comments=True,
            deny_multiple_statements=True,
            deny_cte=False,
            deny_set_ops=True,
            deny_subqueries=False,
        )
    )

    validation = run_validation(trace_id=trace_id, generation=generation, guard=guard)
    final = build_final_answer(trace_id=trace_id, generation=generation)

    return {
        "analysis": analysis,
        "strategy": strategy,
        "generation": generation,
        "validation": validation,
        "final": final,
    }
