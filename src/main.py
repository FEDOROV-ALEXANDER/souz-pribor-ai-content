"""FastAPI endpoint для генерации карточки товара.

Запуск: uv run uvicorn src.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .llm_client import LLMClient, load_settings
from .pipeline import generate_card
from .schemas import CardDraft, ProductInput

app = FastAPI(title="AI-помощник карточек КИП", version="0.1.0")
_llm = LLMClient()


@app.get("/health")
def health() -> dict[str, str]:
    settings = load_settings()
    return {"status": "ok", "model": settings.llm_model, "base_url": settings.llm_base_url}


@app.post("/generate", response_model=CardDraft)
def generate(product: ProductInput) -> CardDraft:
    try:
        return generate_card(product, _llm)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
