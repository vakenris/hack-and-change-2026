"""Точка входа платформы: читает questions.csv из DATA_DIR и пишет submission.csv."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant_core.assistant_service import build_assistant  # noqa: E402
from assistant_core.contracts import Answer, Message, TechnicalError  # noqa: E402
from assistant_core.runtime import deadline as request_deadline, timings  # noqa: E402

FALLBACK = "Не удалось сформировать ответ из-за технической ошибки. Вывод по вопросу не сделан."
MAX_ANSWER_CHARS = 10_000
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


DECISIVE_WORDING = re.compile(
    r"\b(не позднее|не ранее|не более|не менее|свыше|за исключением|"
    r"только при|при условии|в случаях, не указанных)(?!\w)(?:[^.!?;\r\n]|\.(?=\d))+",
    re.IGNORECASE,
)


def decisive_excerpt(quote: str, claim: str) -> str:
    claim_lower = claim.casefold()
    claim_numbers = set(re.findall(r"\d+", claim))
    for match in DECISIVE_WORDING.finditer(quote):
        excerpt = match[0].strip(" ,")
        if len(excerpt) > 140 or excerpt.casefold() in claim_lower:
            continue
        # Skip incomplete references such as an excerpt ending at "п.".
        if re.match(r"\.\s*\d", quote[match.end():]):
            continue
        if match[1].casefold() in claim_lower or claim_numbers.intersection(re.findall(r"\d+", excerpt)):
            return excerpt
    return ""


def format_answer(answer: Answer) -> str:
    """Render verified conclusions with inline references and selective quotations."""
    note = answer.note.strip() if answer.claims else answer.text.strip()
    if answer.missing_fact and answer.missing_fact not in answer.text:
        note = "\n".join(filter(None, (note, f"Не хватает сведений: {answer.missing_fact}.")))
    by_id = {source.fragment_id: source for source in answer.sources}
    blocks, quotes = [], []
    omitted = False
    for claim in answer.claims:
        labels, excerpt = [], ""
        for fragment_id in claim.fragment_ids:
            source = by_id.get(fragment_id)
            if source is None:
                continue
            label = f"{source.document_title}, {source.clause}"
            if source.revision_label:
                label += f" ({source.revision_label})"
            if label not in labels:
                labels.append(label)
            if not excerpt:
                excerpt = decisive_excerpt(source.quote, claim.text)
        if labels:
            block = f"{claim.text.strip()} [{'; '.join(labels)}]"
            if sum(len(part) + 2 for part in blocks) + len(block) + len(note) + 80 <= MAX_ANSWER_CHARS:
                blocks.append(block)
                quotes.append(excerpt)
                continue
        omitted = True
    if omitted and blocks:
        note = "\n".join(filter(None, (note, "Часть выводов опущена из-за ограничения длины ответа.")))
    if answer.claims and not blocks:
        return FALLBACK
    spare = MAX_ANSWER_CHARS - len("\n\n".join(filter(None, (*blocks, note))))
    used_quotes = set()
    for i, excerpt in enumerate(quotes):
        if excerpt and excerpt not in used_quotes:
            quoted = f" «{excerpt}»"
            if len(quoted) <= spare:
                blocks[i] += quoted
                spare -= len(quoted)
                used_quotes.add(excerpt)
    text = "\n\n".join(filter(None, (*blocks, note)))
    return text if 0 < len(text) <= MAX_ANSWER_CHARS else FALLBACK


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
        rows = [(row[id_column], row[question_column] or "") for row in reader]
    ids = [row_id for row_id, _ in rows]
    if any(not row_id or not row_id.strip() for row_id in ids) or len(ids) != len(set(ids)):
        raise SystemExit("В questions.csv обнаружены пустые или повторяющиеся ID")
    return rows


@dataclass
class RunResult:
    text: str = FALLBACK
    status: str = "error"
    stage_ms: dict[str, int] = field(default_factory=dict)
    failed_stage: str | None = None
    code: str | None = None


def answer_one(raw: str, deadline: float) -> RunResult:
    """Один тест-кейс. История между кейсами не передаётся."""
    if time.monotonic() >= deadline:
        return RunResult(failed_stage="batch", code="deadline")
    stages: dict[str, int] = {}
    end_token = request_deadline.set(deadline)
    timing_token = timings.set(stages)
    try:
        if not hasattr(_local, "assistant"):
            packaged_db = ROOT / "knowledge.sqlite"
            db_path = os.environ.get("KNOWLEDGE_DB") or (
                str(packaged_db) if packaged_db.is_file() else None
            )
            _local.assistant = build_assistant(db_path)
        answer = _local.assistant.answer(parse_dialogue(raw))
        text = format_answer(answer)
        if text == FALLBACK:
            return RunResult(stage_ms=stages, failed_stage="format", code="output_length")
        return RunResult(text, answer.status, stages)
    except TechnicalError as error:
        return RunResult(stage_ms=stages, failed_stage=error.stage or "startup", code=error.code)
    except ValueError:
        return RunResult(stage_ms=stages, failed_stage="input", code="invalid_question")
    except Exception as error:
        return RunResult(stage_ms=stages, failed_stage="batch", code=type(error).__name__)
    finally:
        request_deadline.reset(end_token)
        timings.reset(timing_token)


def run_batch(rows: list[tuple[str, str]], deadline: float, workers: int) -> list[RunResult]:
    pending = Queue()
    done = Queue()
    for index, (_, raw) in enumerate(rows):
        pending.put((index, raw))

    def work() -> None:
        while time.monotonic() < deadline:
            try:
                index, raw = pending.get_nowait()
            except Empty:
                return
            result = answer_one(raw, deadline)
            if time.monotonic() <= deadline:
                done.put((index, result))

    # A stalled request must not keep the CLI alive after the batch deadline.
    for _ in range(min(workers, len(rows))):
        threading.Thread(target=work, daemon=True).start()

    results = [RunResult(failed_stage="batch", code="deadline") for _ in rows]
    for _ in rows:
        try:
            index, result = done.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty:
            break
        results[index] = result
    return results


def write_submission(output: Path, rows: list[tuple[str, str]], results: list[RunResult]) -> None:
    if len(rows) != len(results) or any(not 0 < len(result.text.strip()) <= MAX_ANSWER_CHARS for result in results):
        raise RuntimeError("Неполные или некорректные результаты сабмита")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as handle:
        staging = Path(handle.name)
        try:
            writer = csv.writer(handle)
            writer.writerow(["id", "answer"])
            writer.writerows((row_id, result.text) for (row_id, _), result in zip(rows, results, strict=True))
            handle.close()
            staging.replace(output)
        finally:
            handle.close()
            staging.unlink(missing_ok=True)


def report_run(rows: list[tuple[str, str]], results: list[RunResult], started: float, workers: int) -> None:
    detailed = os.environ.get("SUBMISSION_DIAGNOSTICS") == "1"
    for (row_id, _), result in zip(rows, results, strict=True):
        if detailed or result.code:
            print(json.dumps({
                "event": "question", "id": row_id, "status": result.status, "stage_ms": result.stage_ms,
                "failed_stage": result.failed_stage, "code": result.code,
            }, ensure_ascii=True), file=sys.stderr)
    print(json.dumps({
        "event": "batch", "rows": len(rows), "workers": workers, "statuses": dict(Counter(r.status for r in results)),
        "errors": dict(Counter(r.code for r in results if r.code)), "elapsed_ms": round((time.monotonic() - started) * 1000),
    }), file=sys.stderr)


def main() -> int:
    started = time.monotonic()
    data_dir = Path(os.environ.get("DATA_DIR") or "data")
    output = Path(os.environ.get("OUTPUT_PATH", "submission.csv"))
    try:
        budget = float(os.environ.get("TIME_BUDGET_SECONDS") or "1080")
        workers = int(os.environ.get("MAX_WORKERS") or os.environ.get("SOLUTION_WORKERS") or "4")
        if not math.isfinite(budget) or budget <= 0 or workers <= 0:
            raise ValueError
    except ValueError:
        raise SystemExit("Некорректны TIME_BUDGET_SECONDS или MAX_WORKERS") from None

    rows = read_questions(data_dir / "questions.csv")
    # Reserve the last 30 seconds of the platform limit for writing and logging.
    results = run_batch(rows, started + min(budget, 1170), workers)
    write_submission(output, rows, results)
    report_run(rows, results, started, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
