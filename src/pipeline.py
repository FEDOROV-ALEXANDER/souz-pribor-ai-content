"""Оркестрация генерации: пред-проверка → LLM → валидация → correction → сборка.

Перенос продового guardrail-паттерна: код проверяет то, что модель сформулировала,
и при ошибках даёт ей один correction-проход с перечнем проблем. После исчерпания
попыток — честный отказ, а не «почти подходящий» результат.
"""

from __future__ import annotations

import json

from .llm_client import LLMClient
from .prompt import SYSTEM_PROMPT, build_correction_prompt, build_user_prompt
from .schemas import CardDraft, GeneratedSections, ProductInput
from .validation import parse_and_validate

MAX_ATTEMPTS = 2  # первая генерация + один correction-проход


def generate_card(product: ProductInput, llm: LLMClient) -> CardDraft:
    product_json = product.model_dump_json(ensure_ascii=False, indent=2)

    warnings: list[str] = []
    missing = product.missing_input_fields()
    if not product.specs:
        warnings.append("specs пуст: ключевые характеристики не сформированы, только текстовые блоки")

    sections, checks, attempts = _generate_with_guardrail(product_json, product, llm)
    if sections is None:
        raise ValueError("Не удалось получить валидный результат от LLM: " + "; ".join(checks))

    # Характеристики собирает код из входа — модель их не переписывает.
    key_specs = dict(product.specs)
    if not product.specs:
        warnings.append("данных нет: технические характеристики не предоставлены")

    return CardDraft(
        seo_title=sections.seo_title,
        seo_description=sections.seo_description,
        key_specs=key_specs,
        search_queries=sections.search_queries,
        commercial_block=sections.commercial_block,
        missing_fields=missing + sections.missing_fields,
        warnings=warnings,
        validation={"checks": checks, "attempts": attempts, "status": "ok"},
    )


def _generate_with_guardrail(
    product_json: str,
    product: ProductInput,
    llm: LLMClient,
) -> tuple[GeneratedSections | None, list[str], int]:
    checks: list[str] = []
    raw = llm.chat(SYSTEM_PROMPT, build_user_prompt(product_json))
    sections, errors = parse_and_validate(raw, product)
    checks.append("генерация: схема и сверка характеристик — ок" if not errors else f"генерация: {len(errors)} проблем")
    if not errors and sections is not None:
        return sections, checks, 1

    # Correction-проход: ошибки уходят модели обратно, как в продовых guardrails.
    # Всего вызовов LLM — не больше MAX_ATTEMPTS (генерация + одна коррекция).
    for attempt in range(2, MAX_ATTEMPTS + 1):
        raw = llm.chat(SYSTEM_PROMPT, build_correction_prompt(product_json, raw, errors))
        sections, errors = parse_and_validate(raw, product)
        checks.append(f"попытка {attempt}: {'ок' if not errors else f'{len(errors)} проблем'}")
        if not errors and sections is not None:
            return sections, checks, attempt
    return None, checks + errors, MAX_ATTEMPTS
