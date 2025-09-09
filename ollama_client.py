import httpx
import json
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate_structured(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Генерация ответа с структурированным выводом"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_ctx": 8192
                    },
                    "keep_alive": "5m",
                }
                
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                
                result = response.json()
                try:
                    raw = result["response"].strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\n", "", raw)
                        raw = re.sub(r"```$", "", raw)
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Некорректный JSON в ответе модели: {result.get('response')}"
                    )
                    raise ValueError("Модель вернула некорректный JSON") from e
                
        except Exception as e:
            logger.error(f"Ошибка генерации с моделью {model}: {e}")
            raise

    async def generate_text(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        """Генерация обычного текстового ответа"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_ctx": 8192,
                    },
                    "keep_alive": "5m",
                }

                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()

                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ошибка генерации текста с моделью {model}: {e}")
            raise

