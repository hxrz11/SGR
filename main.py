import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from database import DatabaseManager
from ollama_client import OllamaClient
from sgr_schema import DATABASE_SCHEMA, EXAMPLE_QUERIES, SQLGeneration

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные объекты
db_manager = DatabaseManager()
ollama_client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for startup and shutdown"""
    await db_manager.initialize()
    logger.info("Приложение запущено")
    try:
        yield
    finally:
        await db_manager.close()


app = FastAPI(title="Text2SQL POC с SGR", version="1.0.0", lifespan=lifespan)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

DEFAULT_MODEL = "qwen3:32b"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class QueryResponse(BaseModel):
    sql_query: str
    explanation: str
    confidence: float
    results: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: int
    model_used: str
    final_answer: str = ""
    steps: List[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница"""
    index_path = BASE_DIR / "static" / "index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/history")
async def get_history():
    """Возврат последних 5 запросов"""
    files = sorted(LOGS_DIR.glob("*.json"), reverse=True)[:5]
    history = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                history.append(json.load(f))
        except Exception as e:
            logger.warning("Не удалось прочитать лог %s: %s", file, e)
    return {"logs": history}


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Обработка естественного запроса"""
    start_time = time.time()

    # Общий шаблон промпта для SGR
    base_prompt = f"""
Ты эксперт по SQL и работе с базами данных. Твоя задача - преобразовать естественный запрос на русском языке в корректный SQL запрос.

СХЕМА БАЗЫ ДАННЫХ:
{DATABASE_SCHEMA}

ПРИМЕРЫ ЗАПРОСОВ:
{chr(10).join(EXAMPLE_QUERIES)}

ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС: "{request.question}"

Следуй Schema-Guided Reasoning подходу:
1. Проанализируй запрос пользователя
2. Определи стратегию построения SQL
3. Сгенерируй корректный SQL запрос
4. Объясни логику

ВАЖНО:
- Используй только SELECT запросы
- Все названия полей в двойных кавычках
- Для текстового поиска используй ILIKE '%term%'
- Для номенклатуры ищи по двум полям: ("Nomenclature" ILIKE '%term%' OR "NomenclatureFullName" ILIKE '%term%')
    - Для поиска пользователей ищи по трём полям: ("UserName" ILIKE '%term%' OR "PurchaseCardUserName" ILIKE '%term%' OR "PurchaseCardUserFio" ILIKE '%term%')
- Не добавляй LIMIT если пользователь явно не просил ограничить результаты
"""

    # Получение схемы для структурированного вывода
    schema = SQLGeneration.model_json_schema()

    logger.info("Model used: %s", DEFAULT_MODEL)

    error_message = None
    last_sql = ""
    steps: List[str] = ["Запрос получен"]

    try:
        for attempt in range(2):
            prompt = base_prompt
            if error_message:
                prompt += f"\nПредыдущий SQL вызвал ошибку: {error_message}\nИсправь запрос с учётом этой ошибки."

            logger.info("Prompt: %s", prompt)

            result = await ollama_client.generate_structured(
                model=DEFAULT_MODEL,
                prompt=prompt,
                schema=schema,
                temperature=0.2,
            )

            try:
                sgr_result = SQLGeneration(**result)
            except ValidationError as e:
                logger.error(f"Ошибка валидации: {e}")
                raise HTTPException(status_code=422, detail=e.errors())

            last_sql = sgr_result.sql_query
            steps.append("SQL запрос сгенерирован")

            try:
                query_results, executed_sql = await db_manager.execute_query(
                    sgr_result.sql_query
                )
            except ValueError as e:
                error_message = str(e)
                logger.warning("SQL execution failed: %s", error_message)
                continue

            execution_time = int((time.time() - start_time) * 1000)
            logger.info(
                "Executed SQL: %s | Execution time: %d ms",
                executed_sql,
                execution_time,
            )
            steps.append("SQL запрос выполнен")
            final_prompt = (
                f'Вопрос пользователя: "{request.question}"\n'
                f"Ответ базы данных в формате JSON:\n{json.dumps(query_results, ensure_ascii=False)}\n\n"
                "Сформулируй лаконичный, грамотный и вежливый ответ на русском языке, используя только эти данные. "
                "Если данных недостаточно, сообщи, что по запросу ничего не найдено."
            )
            final_answer = await ollama_client.generate_text(
                model=DEFAULT_MODEL,
                prompt=final_prompt,
                temperature=0.2,
            )
            steps.append("Ответ сформирован")

            response = QueryResponse(
                sql_query=executed_sql,
                explanation=sgr_result.explanation,
                confidence=sgr_result.confidence_score,
                results=query_results,
                execution_time_ms=execution_time,
                model_used=DEFAULT_MODEL,
                final_answer=final_answer,
                steps=steps,
            )

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": request.question,
                "sql_query": executed_sql,
                "raw_response": query_results,
                "results": query_results,
                "explanation": sgr_result.explanation,
                "confidence": sgr_result.confidence_score,
                "execution_time_ms": execution_time,
                "model_used": DEFAULT_MODEL,
                "final_answer": final_answer,
                "steps": steps,
            }
            log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            try:
                log_data = jsonable_encoder(log_entry)
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Не удалось сохранить лог: %s", e)

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    execution_time = int((time.time() - start_time) * 1000)
    return QueryResponse(
        sql_query=last_sql,
        explanation=f"Не удалось выполнить запрос: {error_message}",
        confidence=0.0,
        results=[],
        execution_time_ms=execution_time,
        model_used=DEFAULT_MODEL,
        final_answer="",
        steps=steps,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
