"""Точка входа платформы: читает questions.csv из DATA_DIR и пишет submission.csv."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant_core.assistant_service import build_assistant  # noqa: E402
from assistant_core.contracts import Answer, Message  # noqa: E402

FALLBACK = "Не удалось сформировать ответ из-за технической ошибки. Вывод по вопросу не сделан."
QUESTION_COLUMNS = ("question", "query", "text", "dialog", "dialogue", "messages", "input")
USER_ROLES = {"пользователь", "user", "human"}
ROLE_RE = re.compile(
    r"^\s*(пользователь|user|human|ассистент|assistant|бот|bot)\s*:\s*", re.IGNORECASE
)
_local = threading.local()


def parse_dialogue(raw: str) -> list[Message]:
    """Одиночный вопрос, JSON-история или строки с метками ролей."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            items = json.loads(raw)
            parsed = [
                Message(
                    "assistant" if str(item.get("role", "")).lower() in ("assistant", "bot") else "user",
                    str(item.get("content", "")),
                )
                for item in items
            ]
            if parsed and parsed[-1].role == "user":
                return parsed
        except (ValueError, AttributeError, TypeError):
            pass

    messages: list[Message] = []
    role, buffer, marked = "user", [], False
    for line in raw.splitlines():
        match = ROLE_RE.match(line)
        if match:
            marked = True
            text = "\n".join(buffer).strip()
            if text:
                messages.append(Message(role, text))
            role = "user" if match.group(1).lower() in USER_ROLES else "assistant"
            buffer = [line[match.end():]]
        else:
            buffer.append(line)
    text = "\n".join(buffer).strip()
    if text:
        messages.append(Message(role, text))
    if not marked or not messages or messages[-1].role != "user":
        return [Message("user", raw)]
    return messages


def format_answer(answer: Answer) -> str:
    """Текст для CSV: вывод, недостающие сведения, названия документов и номера положений."""
    parts = [answer.text.strip()]
    if answer.missing_fact and answer.missing_fact not in answer.text:
        parts.append("Не хватает сведений: " + answer.missing_fact + ".")
    grounds: list[str] = []
    for source in answer.sources:
        label = f"{source.document_title}, {source.clause}"
        if source.revision_label:
            label += f" ({source.revision_label})"
        if label not in grounds:
            grounds.append(label)
    if grounds:
        parts.append("Основания: " + "; ".join(grounds) + ".")
    return "\n".join(parts)


def read_questions(path: Path) -> list[tuple[str, str]]:
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        lowered = {name.strip().lower(): name for name in fields}
        id_column = lowered.get("id", fields[0] if fields else None)
        question_column = next((lowered[c] for c in QUESTION_COLUMNS if c in lowered), None)
        if question_column is None:
            question_column = next((name for name in fields if name != id_column), None)
        if id_column is None or question_column is None:
            raise SystemExit(f"Не удалось определить столбцы во входном файле: {fields}")
        return [(row[id_column], row[question_column] or "") for row in reader]


def answer_one(raw: str, deadline: float) -> str:
    """Один тест-кейс. История между кейсами не передаётся."""
    if time.monotonic() > deadline:
        return FALLBACK
    try:
        if not hasattr(_local, "assistant"):
            _local.assistant = build_assistant()
        text = format_answer(_local.assistant.answer(parse_dialogue(raw)))
        return text if text.strip() else FALLBACK
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return FALLBACK


def main() -> int:
    data_dir = Path(os.environ.get("DATA_DIR", "."))
    output = Path(os.environ.get("OUTPUT_PATH", "submission.csv"))
    budget = float(os.environ.get("TIME_BUDGET_SECONDS", "1080"))
    workers = max(1, int(os.environ.get("SOLUTION_WORKERS", "4")))

    rows = read_questions(data_dir / "questions.csv")
    deadline = time.monotonic() + budget
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda row: answer_one(row[1], deadline), rows))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        for (row_id, _), text in zip(rows, answers):
            writer.writerow([row_id, text])
    print(f"Готово: {len(rows)} ответов в {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())