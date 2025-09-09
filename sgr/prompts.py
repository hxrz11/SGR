ANALYSIS_SYSTEM = (
    "You are the ANALYSIS stage of SGR. Extract intent, entities, filters, "
    "constraints, and ambiguity flags. Respond strictly as JSON matching AnalysisResult."
)

STRATEGY_SYSTEM = (
    "You are the STRATEGY stage of SGR. Choose a query pattern, explain rationale and risks, "
    "and pick a model. Respond strictly as JSON matching StrategyPlan."
)

GENERATION_SYSTEM = (
    "You are the GENERATION stage of SGR. Produce parameterized SQL (no literals from user input), "
    "explain rationale, and list expected columns. Respond strictly as JSON matching GenerationResult."
)

VALIDATION_SYSTEM = (
    "You are the VALIDATION stage of SGR. Summarize static safety checks and execution outcome. "
    "Respond strictly as JSON matching ValidationReport."
)

# Примеры user->assistant JSON можно добавить как few-shot, но держите ответ строго JSON.
