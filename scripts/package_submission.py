"""Собирает ZIP для платформы: python scripts/package_submission.py"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 128 * 1024 * 1024

CORE_MODULES = [
    "__init__.py",
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
]
# пары: путь в репозитории, путь в архиве
FILES = [("solution.py", "solution.py"), ("submission-requirements.txt", "requirements.txt")]
FILES += [(f"assistant_core/{name}", f"assistant_core/{name}") for name in CORE_MODULES]
FILES += [
    ("data/processed/knowledge.sqlite", "data/processed/knowledge.sqlite"),
    ("data/processed/manifest.json", "data/processed/manifest.json"),
]


def build_zip(output: Path, strict: bool = True) -> Path:
    missing = [source for source, _ in FILES if not (ROOT / source).is_file()]
    if missing and strict:
        raise SystemExit("Нет файлов для архива: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, target in FILES:
            if (ROOT / source).is_file():
                archive.write(ROOT / source, target)
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