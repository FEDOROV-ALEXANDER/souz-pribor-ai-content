"""Пост-валидация ответа LLM — guardrail против выдуманных характеристик.

Слои защиты:
1. Схема и ограничения длин (Pydantic + явные проверки).
2. Сверка технических значений: каждое значение из входного specs обязано
   встретиться в seo_title/seo_description (кроме общих меток вроде "мм" —
   для них проверяем единицы измерения).
3. Запрещённые маркетинговые клише.
Ошибки возвращаются списком — пайплайн решает: correction-проход или отказ.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .schemas import GeneratedSections, ProductInput

TITLE_MAX = 70
DESC_MIN = 800
DESC_MAX = 1200
BANNED_CLICHES = ("лучш", "идеальн", "гарантированн", "уникальн")

# Общие единицы, которые часто входят в любые характеристики и не доказывают выдумку.
GENERIC_UNITS = {"мм", "мПа", "МПа", "кПа", "Па", "бар", "мА", "В", "кг", "г", "°C"}


def _split_number_unit(value: str) -> tuple[str, str] | None:
    """Отделяет числовую часть от единицы измерения: '0-1.6 МПа' -> ('0-1.6', 'МПа')."""
    match = re.match(r"^\s*([\-−–—0-9.,/x×]+)\s*(.*)$", value)
    if not match:
        return None
    number, unit = match.group(1), match.group(2).strip()
    if not unit:
        return None
    return number, unit


def _spec_value_present(raw_value: str, haystack: str) -> bool:
    """Ищет значение характеристики в тексте с учётом нормализации и формы записи.

    Для значений вида «число + единица» требует, чтобы число и единица встречались
    рядом — иначе переписанное число («0–2,5 МПа») пройдёт проверку по одной
    общей единице («МПа»). Это закрывает ложноотрицательный случай подмены числа.
    """
    variants = _value_variants(raw_value)
    if any(variant in haystack for variant in variants):
        return True
    parsed = _split_number_unit(_normalize_value(raw_value))
    if parsed is None:
        return False
    number, unit = parsed
    if number not in haystack:
        return False
    number_pos = haystack.find(number)
    unit_pos = haystack.find(unit, number_pos)
    return 0 <= unit_pos - number_pos <= len(number) + 2


def extract_json_object(text: str) -> dict:
    """Вырезает первый JSON-объект из ответа модели, допуская markdown-обёртку."""
    start = text.find("{")
    if start == -1:
        raise ValueError("В ответе модели нет JSON-объекта")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("JSON-объект в ответе модели не закрыт")


def _normalize_value(raw: str) -> str:
    """Нормализация: тире/дефис → '-', запятая-десятичный разделитель → '.'."""
    value = raw.replace("–", "-").replace("—", "-").replace("−", "-")
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    return value


def _value_variants(raw: str) -> list[str]:
    """Варианты написания значения: исходный и нормализованный, если отличаются."""
    normalized = _normalize_value(raw)
    variants = [raw]
    if normalized != raw:
        variants.append(normalized)
    return variants


def validate_generated(
    generated: GeneratedSections,
    product: ProductInput,
) -> list[str]:
    """Проверяет ответ модели. Возвращает список ошибок (пусто — ок)."""
    errors: list[str] = []
    haystack = f"{generated.seo_title}\n{generated.seo_description}"

    if len(generated.seo_title) > TITLE_MAX:
        errors.append(f"seo_title длиннее {TITLE_MAX} символов ({len(generated.seo_title)})")
    if not (DESC_MIN <= len(generated.seo_description) <= DESC_MAX):
        errors.append(
            f"seo_description вне диапазона {DESC_MIN}-{DESC_MAX} символов "
            f"({len(generated.seo_description)})"
        )

    for phrase in BANNED_CLICHES:
        if phrase in haystack.lower():
            errors.append(f"запрещённое клише: «{phrase}…»")

    for key, raw_value in product.specs.items():
        if not _spec_value_present(raw_value, haystack):
            errors.append(
                f"в seo-полях не найдено значение характеристики «{key}: {raw_value}» "
                f"— либо упущено, либо изменено"
            )
    return errors


def parse_and_validate(raw_text: str, product: ProductInput) -> tuple[GeneratedSections | None, list[str]]:
    """Парсит сырой ответ модели и валидирует. Возвращает (секции, ошибки)."""
    try:
        data = extract_json_object(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, [f"ответ модели — не корректный JSON: {exc}"]
    try:
        sections = GeneratedSections.model_validate(data)
    except ValidationError as exc:
        return None, [f"ответ модели не соответствует схеме: {exc.errors()[:3]}"]
    errors = validate_generated(sections, product)
    return sections, errors
