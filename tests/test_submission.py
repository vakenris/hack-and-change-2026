import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.package_submission import build_zip

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


def run_solution(workdir: Path, tmp_path: Path) -> dict:
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


def check(answers: dict) -> None:
    assert all(text.strip() for text in answers.values())
    assert "склад № 2" in answers["q1"]
    assert "Договор поставки № П-17/2024, п. 3.2" in answers["q1"]
    assert "Уточните" in answers["q2"]  # дата из q1 не перенеслась в q2
    assert "склад № 1" in answers["q3"]  # многострочный диалог разобран
    assert "технической ошибки" in answers["q4"]
    assert "F-999" not in "".join(answers.values())


def test_csv_format_and_independence(tmp_path):
    check(run_solution(ROOT, tmp_path))


def test_runs_from_unpacked_zip(tmp_path):
    archive = build_zip(tmp_path / "submission.zip", strict=False)
    target = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        handle.extractall(target)
    assert "solution.py" in names
    assert "requirements.txt" in names
    check(run_solution(target, tmp_path))