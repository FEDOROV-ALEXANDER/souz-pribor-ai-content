"""CLI: генерация черновика карточки из JSON-файла.

Примеры:
    python -m src.cli data/product_1.json
    python -m src.cli data/product_1.json --format markdown
    python -m src.cli data --all --format markdown --out examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .llm_client import LLMClient
from .pipeline import generate_card
from .schemas import CardDraft, ProductInput

PROVIDED_EXAMPLE = "product_1"
FIXTURE_NOTE = (
    "> Вход `{file}` — демонстрационный fixture по той же схеме: в задании приложен "
    "один пример структуры (манометр), второй товар не предоставлен.\n"
)


def _read_product(path: Path) -> ProductInput:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Файл не найден: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Некорректный JSON в {path}: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"В {path} ожидается JSON-объект товара")
    try:
        return ProductInput.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"Вход не соответствует схеме: {exc.errors()[:3]}")


def to_markdown(card: CardDraft, source: Path) -> str:
    lines: list[str] = [f"# {card.seo_title}", ""]
    if source.stem != PROVIDED_EXAMPLE:
        lines += [FIXTURE_NOTE.format(file=source.name), ""]
    lines += [
        "## SEO-описание",
        "",
        card.seo_description,
        "",
        "## Ключевые характеристики",
        "",
    ]
    if card.key_specs:
        lines += [f"- **{key}**: {value}" for key, value in card.key_specs.items()]
    else:
        lines.append("данных нет: технические характеристики не предоставлены")
    lines += ["", "## Поисковые запросы", ""]
    lines += [f"{i}. {query}" for i, query in enumerate(card.search_queries, start=1)]
    lines += [
        "",
        "## Коммерческий блок",
        "",
        card.commercial_block,
        "",
        "## Чего не хватает",
        "",
    ]
    lines += [f"- {item}" for item in card.missing_fields] if card.missing_fields else ["- нет"]
    lines += ["", "## Проверки", ""]
    lines += [f"- {check}" for check in card.validation.get("checks", [])]
    if card.warnings:
        lines += ["", "## Предупреждения", ""]
        lines += [f"- {warning}" for warning in card.warnings]
    return "\n".join(lines) + "\n"


def _render(card: CardDraft, source: Path, fmt: str) -> str:
    if fmt == "markdown":
        return to_markdown(card, source)
    return card.model_dump_json(ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Генератор черновика карточки товара КИП")
    parser.add_argument("input", help="JSON-файл товара или каталог data/ с флагом --all")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", help="Каталог для результатов (при --all) или файл")
    parser.add_argument("--all", action="store_true", help="Прогнать все *.json из входа")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if args.all:
        if not path.is_dir():
            raise SystemExit("С флагом --all вход должен быть каталогом")
        sources = sorted(path.glob("*.json"))
        if not sources:
            raise SystemExit(f"В {path} нет JSON-файлов")
    else:
        sources = [path]

    llm = LLMClient()
    for source in sources:
        product = _read_product(source)
        try:
            card = generate_card(product, llm)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"Ошибка генерации для {source}: {exc}")
        rendered = _render(card, source, args.format)
        if args.all:
            out_dir = Path(args.out) if args.out else Path("examples")
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "md" if args.format == "markdown" else "json"
            target = out_dir / f"{source.stem}_result.{suffix}"
            target.write_text(rendered, encoding="utf-8")
            print(f"готово: {source} -> {target}")
        else:
            sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
