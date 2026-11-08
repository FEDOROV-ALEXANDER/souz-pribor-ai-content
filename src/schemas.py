"""Pydantic-схемы входа и выхода сервиса.

Вход описывает товар КИП, выход — черновик карточки для контент-менеджера.
Схема выхода фиксирует то, что требует задание: seo_title, seo_description,
key_specs, search_queries, commercial_block, missing_fields, warnings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    """Входные данные о товаре. Допускаем лишние поля — не ломаемся на них."""

    model_config = {"extra": "allow"}

    name: str | None = Field(default=None, description="Название товара")
    manufacturer: str | None = Field(default=None, description="Производитель")
    specs: dict[str, str] = Field(default_factory=dict)
    description: str | None = Field(default=None, description="Текст из паспорта/описания")

    def missing_input_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.name or not self.name.strip():
            missing.append("name")
        if not self.manufacturer or not self.manufacturer.strip():
            missing.append("manufacturer")
        if not self.specs:
            missing.append("specs")
        if not self.description or not self.description.strip():
            missing.append("description")
        return missing


class GeneratedSections(BaseModel):
    """Та часть результата, которую формулирует LLM (характеристики — не она)."""

    seo_title: str
    seo_description: str
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial_block: str
    missing_fields: list[str] = Field(default_factory=list)


class CardDraft(BaseModel):
    """Итоговый черновик карточки товара."""

    seo_title: str
    seo_description: str
    key_specs: dict[str, str]
    search_queries: list[str]
    commercial_block: str
    missing_fields: list[str]
    warnings: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(
        default_factory=dict,
        description="Прозрачный отчёт о проверках: что проверяли, итог, попытки",
    )
