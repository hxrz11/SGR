import json
import logging
import os
import time
import asyncio
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
from agent_tools import run_sql, call_status_api

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
Ты — аналитик по закупочной базе данных. Ты решаешь задачи пользователя, используя пошаговое рассуждение и доступные действия.

Доступные действия:
- run_sql(sql_query): выполняет только SELECT запросы и возвращает строки результата.
- call_status_api(purchase_number): возвращает текущий статус обработки закупки по её номеру.
- clarification(question): задаёт пользователю уточняющий вопрос при недостатке данных.
- report_completion(answer, steps): сообщает итоговый ответ и перечисляет выполненные действия.

СХЕМА БАЗЫ ДАННЫХ:
{DATABASE_SCHEMA}

ПРИМЕРЫ ЗАПРОСОВ:
{chr(10).join(EXAMPLE_QUERIES)}

ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС: "{request.question}"

Следуй Schema-Guided Reasoning подходу:
1. Проанализируй запрос пользователя.
2. Составь стратегию построения SQL.
3. При необходимости используй run_sql для получения данных.
4. Если нужен статус обработки, используй call_status_api.
5. Если информации недостаточно, используй clarification.
6. После завершения обязательно вызови report_completion.

ВАЖНЫЕ ПРАВИЛА:
- Используй только SELECT запросы.
- Все названия полей в двойных кавычках.
- Для текстового поиска используй ILIKE '%term%'.
- Для номенклатуры ищи по двум полям: (\"Nomenclature\" ILIKE '%term%' OR \"NomenclatureFullName\" ILIKE '%term%').
- Для поиска пользователей ищи по трём полям: (\"UserName\" ILIKE '%term%' OR \"PurchaseCardUserName\" ILIKE '%term%' OR \"PurchaseCardUserFio\" ILIKE '%term%').
- Не добавляй LIMIT, если пользователь явно не просил ограничить результаты.
- Даты указывай в формате YYYY-MM-DD.
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
                query_results, executed_sql = await run_sql(
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

            purchase_ids = [row.get("PurchaseCardId") for row in query_results if row.get("PurchaseCardId")]
            if purchase_ids:
                status_map: Dict[str, Dict[str, Any]] = {}
                for i in range(0, len(purchase_ids), 20):
                    batch = purchase_ids[i : i + 20]
                    responses = await asyncio.gather(
                        *[call_status_api(pid) for pid in batch],
                        return_exceptions=True,
                    )
                    for pid, resp in zip(batch, responses):
                        if isinstance(resp, Exception):
                            logger.warning("Status API failed for %s: %s", pid, resp)
                            continue
                        timeline = (
                            resp.get("status_timeline")
                            or resp.get("timeline")
                            or resp.get("history")
                            or resp.get("statuses")
                            or []
                        )
                        current_status = None
                        last_date = None
                        if isinstance(timeline, list) and timeline:
                            latest = max(
                                timeline,
                                key=lambda x: x.get("date")
                                or x.get("timestamp")
                                or "",
                            )
                            current_status = latest.get("status") or latest.get("Status")
                            last_date = latest.get("date") or latest.get("timestamp")
                        else:
                            current_status = resp.get("current_status") or resp.get("status")
                            last_date = (
                                resp.get("last_status_date")
                                or resp.get("date")
                                or resp.get("timestamp")
                            )
                        status_map[pid] = {
                            "current_status": current_status,
                            "last_status_date": last_date,
                        }
                for row in query_results:
                    pid = row.get("PurchaseCardId")
                    if pid in status_map:
                        row.update(status_map[pid])
                steps.append("Статусы закупок получены")
            else:
                steps.append("PurchaseCardId не найден; статусы не получены")

            final_prompt = (
                f'Вопрос пользователя: "{request.question}"\n'
                f"Ответ базы данных в формате JSON:\n{json.dumps(query_results, ensure_ascii=False)}\n\n"
                "Сформулируй лаконичный, грамотный и вежливый ответ на русском языке, используя только эти данные. "
                "Если данных недостаточно, сообщи, что по запросу ничего не найдено."
            )
            if not purchase_ids:
                final_prompt += (
                    "\nПоле PurchaseCardId отсутствует, поэтому статусы закупок не получены."
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
