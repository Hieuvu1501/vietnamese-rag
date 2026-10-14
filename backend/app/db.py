import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'processing',
    num_chunks INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    top_chunk_ids_json TEXT,
    triad_json TEXT,
    timings_json TEXT,
    graph_facts_json TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        # messages.timings_json didn't exist in earlier versions of this
        # schema - CREATE TABLE IF NOT EXISTS is a no-op on an existing
        # table, so add it here if missing rather than requiring a fresh DB.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        if "timings_json" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN timings_json TEXT")
        if "graph_facts_json" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN graph_facts_json TEXT")


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def create_document(filename: str, file_type: str) -> int:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO documents (filename, file_type, status) VALUES (?, ?, 'processing')",
            (filename, file_type),
        )
        return cur.lastrowid


def mark_document_done(document_id: int, num_chunks: int) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = 'indexed', num_chunks = ? WHERE id = ?",
            (num_chunks, document_id),
        )


def mark_document_failed(document_id: int, error: str) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = 'failed', error = ? WHERE id = ?",
            (error, document_id),
        )


def fail_orphaned_processing_documents() -> int:
    """
    Uploads are handled synchronously, so a row still marked 'processing' when
    the server starts can only mean the previous process died mid-request -
    never a genuinely in-flight upload. Clears those so they don't sit stuck
    in the UI forever.
    """
    with db_cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = 'failed', error = 'Server restarted while processing this upload' "
            "WHERE status = 'processing'"
        )
        return cur.rowcount


def insert_chunks(document_id: int, chunks: List[Any]) -> None:
    with db_cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, document_id, text, chunk_index, metadata_json) VALUES (?, ?, ?, ?, ?)",
            [
                (c.id, document_id, c.text, c.metadata.get("chunk_index", 0), json.dumps(c.metadata))
                for c in chunks
            ],
        )



def list_documents() -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM documents ORDER BY id DESC")
        return [dict(row) for row in cur.fetchall()]


def delete_document(document_id: int) -> None:
    with db_cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def all_chunks_raw() -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT chunks.id, chunks.text, chunks.metadata_json, documents.filename
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE documents.status = 'indexed'
            ORDER BY chunks.document_id, chunks.chunk_index
            """
        )
        return [dict(row) for row in cur.fetchall()]


def insert_triples(document_id: int, triples: List[Any]) -> None:
    if not triples:
        return
    with db_cursor() as cur:
        cur.executemany(
            "INSERT INTO triples (document_id, chunk_id, subject, predicate, object) VALUES (?, ?, ?, ?, ?)",
            [(document_id, t.chunk_id, t.subject, t.predicate, t.object) for t in triples],
        )


def all_triples_raw() -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT triples.chunk_id, triples.subject, triples.predicate, triples.object,
                   triples.document_id, documents.filename
            FROM triples JOIN documents ON documents.id = triples.document_id
            WHERE documents.status = 'indexed'
            """
        )
        return [dict(row) for row in cur.fetchall()]



def insert_message(
    role: str,
    content: str,
    top_chunk_ids: Optional[List[str]] = None,
    triad: Optional[Dict[str, float]] = None,
    timings: Optional[Dict[str, float]] = None,
    graph_facts: Optional[List[str]] = None,
) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages
                (role, content, top_chunk_ids_json, triad_json, timings_json, graph_facts_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                role,
                content,
                json.dumps(top_chunk_ids) if top_chunk_ids is not None else None,
                json.dumps(triad) if triad is not None else None,
                json.dumps(timings) if timings is not None else None,
                json.dumps(graph_facts) if graph_facts is not None else None,
            ),
        )
        return cur.lastrowid


def list_messages() -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM messages ORDER BY id ASC")
        return [dict(row) for row in cur.fetchall()]
