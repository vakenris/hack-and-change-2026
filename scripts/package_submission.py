"""Собирает ZIP для платформы: python scripts/package_submission.py"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 128 * 1024 * 1024

CORE_MODULES = [
    "contracts.py",
    "assistant_service.py",
    "citations.py",
    "stubs.py",
    "knowledge_store.py",
    "retrieval.py",
    "versions.py",
    "llm_client.py",
    "llm_prompts.py",
    "answer_policy.py",
    "runtime.py",
]
PACKAGE_ENTRY = '''"""Expose the flat submission modules as the assistant_core package."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
'''


def build_zip(output: Path, strict: bool = True, *, database: Path | None = None) -> Path:
    if database is None:
        database = ROOT / "data/processed/knowledge.sqlite"
    files = [
        (ROOT / "solution.py", "solution.py"),
        (ROOT / "submission-requirements.txt", "requirements.txt"),
        (database, "knowledge.sqlite"),
    ]
    files.extend((ROOT / "assistant_core" / name, name) for name in CORE_MODULES)
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing and strict:
        raise SystemExit("Нет файлов для архива: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("assistant_core.py", PACKAGE_ENTRY)
        for source, target in files:
            if source.is_file():
                archive.write(source, target)
    size = output.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"Архив {size / 1024 / 1024:.1f} МБ больше лимита 128 МБ")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "dist" / "submission.zip"))
    parser.add_argument("--allow-missing", action="store_true", help="собрать без отсутствующих файлов")
    args = parser.parse_args()
    archive = build_zip(Path(args.output), strict=not args.allow_missing)
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            print(name)
    print(f"{archive}: {archive.stat().st_size / 1024 / 1024:.2f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
