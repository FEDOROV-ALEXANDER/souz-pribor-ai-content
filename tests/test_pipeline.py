"""Тесты пост-валидации и пайплайна. LLM замокана — без сети."""

from __future__ import annotations

import json

import pytest

from src.pipeline import generate_card
from src.schemas import ProductInput
from src.validation import extract_json_object, parse_and_validate, validate_generated
from src.schemas import GeneratedSections

PRODUCT = ProductInput(
    name="Манометр показывающий DM 02",
    manufacturer="Манотомь",
    specs={
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    description="Прибор предназначен для измерения избыточного давления.",
)

VALID = GeneratedSections(
    seo_title="Манометр показывающий DM 02 Манотомь 0–1,6 МПа",
    seo_description=(
        "Манометр показывающий DM 02 производства Манотомь предназначен для измерения "
        "избыточного давления. Диапазон измерений 0–1,6 МПа, класс точности 1,5. "
        "Присоединение к процессу — резьба G1/2, диаметр корпуса 100 мм. Прибор применяется "
        "в системах контроля давления на промышленных объектах: в трубопроводах, "
        "компрессорных установках, тепловых узлах и технологических линиях, где требуется "
        "наглядный контроль рабочего давления оборудования персоналом без сложной "
        "наладки. Корпус диаметром 100 мм удобен для считывания показаний с рабочего места. "
        "Прямое присоединение G1/2 упрощает монтаж в стандартные отводы. Класс точности 1,5 "
        "соответствует типовым задачам визуального контроля давления в коммунальном и "
        "промышленном теплоснабжении. Прибор подходит для эксплуатации в составе "
        "установок с неагрессивными средами при штатных условиях объекта. Перед публикацией "
        "карточки контент-менеджеру следует уточнить материал корпуса, степень защиты, "
        "рабочую температуру и наличие сертификатов, поскольку эти данные во входном "
        "описании не приведены."
    ),
    search_queries=[
        "манометр показывающий DM 02",
        "манометр 0–1,6 МПа G1/2",
        "манометр Манотомь 100 мм",
    ],
    commercial_block="Подходит инженерам КИПиА и снабженцам для подбора прибора визуального контроля давления.",
    missing_fields=["цена", "материал корпуса"],
)


def _dump(sections: GeneratedSections) -> str:
    return sections.model_dump_json()


class FakeLLM:
    """Скриптованный LLM: выдаёт ответы из очереди."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("LLM вызван чаще, чем ожидалось")
        return self.responses.pop(0)


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_wrapped_in_markdown(self):
        text = 'Ответ:\n```json\n{"a": 1}\n```\nконец'
        assert extract_json_object(text) == {"a": 1}

    def test_no_object(self):
        with pytest.raises(ValueError):
            extract_json_object("нет json")

    def test_unclosed(self):
        with pytest.raises(ValueError):
            extract_json_object('{"a": 1')


class TestValidation:
    def test_valid_passes(self):
        assert validate_generated(VALID, PRODUCT) == []

    def test_title_too_long(self):
        bad = VALID.model_copy(update={"seo_title": "x" * 71})
        errors = validate_generated(bad, PRODUCT)
        assert any("seo_title" in e for e in errors)

    def test_description_too_short(self):
        bad = VALID.model_copy(update={"seo_description": "коротко"})
        errors = validate_generated(bad, PRODUCT)
        assert any("seo_description" in e for e in errors)

    def test_banned_cliche(self):
        bad = VALID.model_copy(update={"seo_title": "Лучший манометр DM 02"})
        errors = validate_generated(bad, PRODUCT)
        assert any("клише" in e for e in errors)

    def test_missing_spec_value_flagged(self):
        # Модель «переписала» диапазон в обоих seo-полях — сверка должна это поймать.
        bad = VALID.model_copy(
            update={
                "seo_title": VALID.seo_title.replace("0–1,6 МПа", "0–2,5 МПа"),
                "seo_description": VALID.seo_description.replace("0–1,6 МПа", "0–2,5 МПа"),
            }
        )
        assert "0–1,6 МПа" not in (bad.seo_title + bad.seo_description)
        errors = validate_generated(bad, PRODUCT)
        assert any("диапазон измерений" in e for e in errors), errors

    def test_broken_json_reported(self):
        sections, errors = parse_and_validate("не json", PRODUCT)
        assert sections is None and errors

    def test_schema_error_reported(self):
        sections, errors = parse_and_validate('{"seo_title": 1}', PRODUCT)
        assert sections is None and any("схеме" in e for e in errors)


class TestPipeline:
    def test_happy_path_single_attempt(self):
        llm = FakeLLM([_dump(VALID)])
        card = generate_card(PRODUCT, llm)
        assert card.key_specs == PRODUCT.specs
        assert card.validation["attempts"] == 1
        assert "name" not in card.missing_fields

    def test_correction_pass_recovers(self):
        broken_title = VALID.model_copy(update={"seo_title": "x" * 71})
        llm = FakeLLM([_dump(broken_title), _dump(VALID)])
        card = generate_card(PRODUCT, llm)
        assert llm.calls == 2
        assert card.validation["attempts"] == 2
        assert len(card.validation["checks"]) == 2

    def test_fails_after_two_attempts(self):
        bad = VALID.model_copy(update={"seo_title": "x" * 71})
        llm = FakeLLM([_dump(bad), _dump(bad)])
        with pytest.raises(ValueError, match="валидный результат"):
            generate_card(PRODUCT, llm)
        assert llm.calls == 2

    def test_empty_specs_warns(self):
        product = PRODUCT.model_copy(update={"specs": {}})
        llm = FakeLLM([_dump(VALID)])
        card = generate_card(product, llm)
        assert card.key_specs == {}
        assert any("specs пуст" in w for w in card.warnings)

    def test_missing_name_reported(self):
        product = PRODUCT.model_copy(update={"name": None})
        llm = FakeLLM([_dump(VALID)])
        card = generate_card(product, llm)
        assert "name" in card.missing_fields
