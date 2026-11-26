import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


class CanonicalDocRegistry:
    """
    Immutable SQLite Canonical Document Registry.
    Persists SHA-256 content hashes and trusted numeric/legal invariants at ingestion
    time so runtime RAG verification never trusts metadata inside untrusted payloads.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(root, "artifacts")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "canonical_registry.db")
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    numeric_invariants_json TEXT NOT NULL,
                    ingested_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    attack_detected INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    neural_score REAL NOT NULL,
                    hash_mismatch INTEGER NOT NULL,
                    numeric_mismatch INTEGER NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_sha256(text: str) -> str:
        normalized = " ".join((text or "").split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        now = time.time()
        with self._connect() as conn:
            for c in chunks:
                sha = self.compute_sha256(c["content"])
                invs = json.dumps(c.get("numeric_invariants", []))
                conn.execute(
                    """
                    INSERT INTO canonical_chunks (chunk_id, doc_id, filename, page, sha256, numeric_invariants_json, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        sha256 = excluded.sha256,
                        numeric_invariants_json = excluded.numeric_invariants_json,
                        ingested_at = excluded.ingested_at
                    """,
                    (c["chunk_id"], c["doc_id"], c["filename"], int(c["page"]), sha, invs, now),
                )
            conn.commit()
        return len(chunks)

    def lookup_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_id, doc_id, filename, page, sha256, numeric_invariants_json FROM canonical_chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "page": row["page"],
                "sha256": row["sha256"],
                "numeric_invariants": json.loads(row["numeric_invariants_json"]),
            }

    def log_security_event(
        self,
        chunk_id: str,
        attack_detected: bool,
        reason: str,
        neural_score: float,
        hash_mismatch: bool,
        numeric_mismatch: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO security_audit_events (chunk_id, attack_detected, reason, neural_score, hash_mismatch, numeric_mismatch, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    int(attack_detected),
                    reason[:240],
                    float(neural_score),
                    int(hash_mismatch),
                    int(numeric_mismatch),
                    time.time(),
                ),
            )
            conn.commit()

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM canonical_chunks").fetchone()
            return int(row["cnt"]) if row else 0


registry = CanonicalDocRegistry()
