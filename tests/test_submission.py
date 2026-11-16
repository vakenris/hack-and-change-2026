import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.build_knowledge import build
from scripts.package_submission import build_zip
from assistant_core.contracts import Answer, Claim, SourceCard
from solution import MAX_ANSWER_CHARS, format_answer

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("q1", "Какой адрес доставки по договору П-17/2024, если спецификация подписана 02.10.2024?"),
    ("q2", "Какой адрес доставки по договору П-17/2024?"),
    (
        "q3",
        "Пользователь: Какой адрес доставки по договору П-17/2024?\n"
        "Ассистент: Уточните дату подписания спецификации.\n"
        "Пользователь: Она подписана 30.09.2024",
    ),
    ("q4", "Проверка: сбой индекса"),
]


def run_solution(workdir: Path, tmp_path: Path) -> dict[str, str]:
    data_dir = tmp_path / "input"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "questions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question"])
        writer.writerows(ROWS)
    output = tmp_path / "submission.csv"
    env = dict(os.environ, DATA_DIR=str(data_dir), OUTPUT_PATH=str(output), ASSISTANT_STUB="1")
    subprocess.run([sys.executable, "solution.py"], cwd=workdir, env=env, check=True, timeout=120)
    with open(output, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == ["id", "answer"]
    assert [row["id"] for row in rows] == [row_id for row_id, _ in ROWS]
    return {row["id"]: row["answer"] for row in rows}


def check(answers: dict[str, str]) -> None:
    assert all(text.strip() for text in answers.values())
    assert "склад № 2" in answers["q1"]
    assert "Договор поставки № П-17/2024, п. 3.2" in answers["q1"]
    assert "Уточните" in answers["q2"]  # дата из q1 не перенеслась в q2
    assert "склад № 1" in answers["q3"]  # многострочный диалог разобран
    assert "технической ошибки" in answers["q4"]
    assert "F-999" not in "".join(answers.values())


def test_csv_format_and_independence(tmp_path: Path) -> None:
    check(run_solution(ROOT, tmp_path))


def test_csv_keeps_each_claim_with_its_own_citation() -> None:
    old = SourceCard("f1", "d1", "Договор", "Редакция 1", "2.1", "Срок оплаты — 5 дней.", 1, 1)
    new = SourceCard("f2", "d2", "Регламент", "Редакция 2", "3.2", "Свыше 15 миллионов.", 2, 2)
    answer = Answer(
        "partial", "Сводный ответ",
        claims=[Claim("Срок — пять дней.", ["f1"]), Claim("Действует порог.", ["f2"]),
                Claim("слишком длинный " * MAX_ANSWER_CHARS, ["f2"])],
        sources=[old, new], note="Порядок уведомления в базе не указан.",
    )
    text = format_answer(answer)
    first, second, note = text.split("\n\n")
    assert "Договор, 2.1 (Редакция 1)" in first and "Регламент" not in first
    assert old.quote in first and new.quote in second
    assert "Порядок уведомления" in note and "ограничения длины" in note
    assert len(text) <= MAX_ANSWER_CHARS


def test_stalled_http_cannot_hold_csv_past_deadline(tmp_path: Path) -> None:
    (tmp_path / "questions.csv").write_text("id,question\n1,быстрый\n2,долгий\n3,ожидающий\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    runner = """
import time
import solution

def answer(raw, deadline):
    if raw == "быстрый":
        return solution.RunResult("Есть подтверждение.", "full")
    time.sleep(20)
    raise AssertionError("the supervisor should have stopped")

solution.answer_one = answer
raise SystemExit(solution.main())
"""
    env = dict(os.environ, DATA_DIR=str(tmp_path), OUTPUT_PATH=str(output), TIME_BUDGET_SECONDS="0.5",
               MAX_WORKERS="1", SOLUTION_WORKERS="3", SUBMISSION_DIAGNOSTICS="1")
    run = subprocess.run([sys.executable, "-c", runner], cwd=ROOT, env=env, check=True,
                         capture_output=True, timeout=8)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert rows[0]["answer"] == "Есть подтверждение."
    assert all("технической ошибки" in row["answer"] for row in rows[1:])
    assert b'"errors": {"deadline": 2}' in run.stderr


def test_runs_from_unpacked_zip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "contract.txt").write_text("ДОГОВОР\n1. Условия\n1.1. Срок оплаты — 5 дней.\n", encoding="utf-8")
    catalogue = tmp_path / "catalogue"
    assert build(raw, catalogue, expected_count=1)["published"]
    archive = build_zip(tmp_path / "submission.zip", database=catalogue / "knowledge.sqlite")
    target = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert all(Path(name).name == name for name in names)
        assert {"solution.py", "requirements.txt", "assistant_core.py", "knowledge.sqlite"} <= set(names)
        handle.extractall(target)

    data_dir = target / "data"
    data_dir.mkdir()
    (data_dir / "questions.csv").write_text("id,question\n1,Какой срок оплаты по п. 1.1?\n", encoding="utf-8")
    env = dict(os.environ, ASSISTANT_STUB="0", LLM_BASE_URL="https://model.example",
               LLM_API_KEY="test-key", LLM_MODEL="test-model", SOLUTION_WORKERS="1")
    for name in ("DATA_DIR", "KNOWLEDGE_DB", "OUTPUT_PATH"):
        env.pop(name, None)

    # Isolated Python can only load the assistant from the unpacked submission.
    runner = """
import json
import runpy

import httpx

def respond(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    data = json.loads(payload["messages"][-1]["content"])
    if "fragments" in data:
        fragment, = data["fragments"]
        assert "Срок оплаты — 5 дней." in fragment["text"]
        result = {"claims": [{"text": "Срок оплаты — 5 дней.", "fragment_ids": [fragment["fragment_id"]]}],
                  "summary": "", "refusal_reason": None}
    elif "claims" in data:
        result = {"supported_indices": [0]}
    else:
        result = {"search_queries": ["срок оплаты"], "document_refs": ["п. 1.1"], "dates": []}
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

with httpx.Client(transport=httpx.MockTransport(respond)) as client:
    httpx.post = client.post
    runpy.run_path("solution.py", run_name="__main__")
"""
    subprocess.run([sys.executable, "-I", "-c", runner], cwd=target, env=env, check=True, timeout=120)
    with (target / "submission.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["id", "answer"]
        row, = list(reader)
    assert row["id"] == "1"
    assert "Срок оплаты — 5 дней." in row["answer"]
    assert "ДОГОВОР, 1.1" in row["answer"]
