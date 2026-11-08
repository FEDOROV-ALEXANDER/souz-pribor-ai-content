"""Тонкий OpenAI-compatible клиент для LLM.

Конфигурация — только из переменных окружения (через .env), ключ никогда
не хранится в коде. Поддерживает любой OpenAI-compatible endpoint: локальный
llama-server/vLLM/SGLang либо облачный API (OpenRouter и т.п.).
"""

from __future__ import annotations

import threading
import time

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_api_key: str = "not-needed-for-local"
    llm_model: str = "qwen3.8-27b-iq4xs"
    llm_timeout_seconds: float = 180.0
    # Минимальная пауза между запросами, сек. Для OpenRouter free-моделей держим >= 4.
    llm_min_request_interval_seconds: float = 0.0


def load_settings() -> LLMSettings:
    """Загружает настройки из cwd/.env. Не печатает секреты."""
    return LLMSettings()


class LLMClient:
    """Синхронный клиент чата с одним повтором по таймауту и паузой между запросами."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or load_settings()
        self._last_request_at: float = 0.0
        self._lock = threading.Lock()

    def _respect_rate_limit(self) -> None:
        interval = self.settings.llm_min_request_interval_seconds
        if interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request_at = time.monotonic()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Один вызов чат-комплишена. Таймаут — один повтор, далее исключение."""
        self._respect_rate_limit()
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    response = client.post(
                        f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                        json=payload,
                    )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    break
        raise RuntimeError(f"LLM недоступна или вернула некорректный ответ: {last_error}")
