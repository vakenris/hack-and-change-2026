"""Read-only access to the SQLite catalogue and its search index."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .contracts import Fragment

SCHEMA_VERSION = 1
FRAGMENT_COLUMNS = (
    "f.fragment_id, f.document_id, f.document_title, f.revision_id, f.revision_label, "
    "f.clause, f.text, f.source_path, f.line_start, f.line_end"
)
CONTRACT_NUMBER = re.compile(r"\b([а-яa-z]+)\s*-\s*(\d+)\s*[/\-]\s*(\d{4})\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return re.sub(r"[‐‑‒–—−]", "-", text)


def search_text(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", normalize(text)))


def contract_numbers(text: str) -> set[str]:
    return {f"{series}-{number}/{year}" for series, number, year in CONTRACT_NUMBER.findall(normalize(text))}


class SQLiteKnowledgeStore:
    def __init__(self, path: Path) -> None:
        self._uri = path.resolve().as_uri() + "?mode=ro"

    def read(self, sql: str, parameters: Sequence[str | int] = ()) -> list[sqlite3.Row]:
        try:
            with closing(sqlite3.connect(self._uri, uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            from .contracts import TechnicalError

            raise TechnicalError(f"Сбой каталога SQLite: {exc}") from exc

    def get_fragment(self, fragment_id: str) -> Fragment | None:
        from .contracts import Fragment

        rows = self.read(
            f"SELECT {FRAGMENT_COLUMNS} FROM fragments f WHERE fragment_id = ?", (fragment_id,)
        )
        return Fragment(**dict(rows[0])) if rows else None

    def get_document_text(self, document_id: str) -> str | None:
        rows = self.read("SELECT text FROM documents WHERE document_id = ?", (document_id,))
        return rows[0]["text"] if rows else None


def open_store(path: str | Path) -> SQLiteKnowledgeStore:
    from .contracts import TechnicalError

    store = SQLiteKnowledgeStore(Path(path))
    if store.read("PRAGMA user_version")[0][0] != SCHEMA_VERSION:
        raise TechnicalError("Неподдерживаемая схема каталога SQLite")
    store.read(f"SELECT {FRAGMENT_COLUMNS} FROM fragments f LIMIT 0")
    store.read("SELECT document_id, text FROM documents LIMIT 0")
    store.read("SELECT family_id, kind, value FROM refs LIMIT 0")
    store.read("SELECT 1 FROM fragment_fts WHERE fragment_fts MATCH 'проверка' LIMIT 1")
    return store
