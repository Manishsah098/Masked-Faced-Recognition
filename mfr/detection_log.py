import sqlite3
import os
from datetime import datetime


class DetectionLog:
    """
    SQLite-backed detection log database.

    Tables
    ------
    persons
        Stores one row per registered/recognised person.
        Columns: id, name, first_seen, last_seen, detection_count

    detection_events
        Fine-grained event log — one row per detection event.
        Columns: id, name, detected_at, mask_status, similarity_score
    """

    def __init__(self, db_path: str = "detection_log.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create tables if they do not already exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    NOT NULL UNIQUE,
                    first_seen      TEXT    NOT NULL,
                    last_seen       TEXT    NOT NULL,
                    detection_count INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS detection_events (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    name             TEXT    NOT NULL,
                    detected_at      TEXT    NOT NULL,
                    mask_status      TEXT    NOT NULL DEFAULT 'Unknown',
                    similarity_score REAL    NOT NULL DEFAULT 0.0
                );
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_detection(
        self,
        name: str,
        mask_status: str = "Unknown",
        similarity_score: float = 0.0,
    ) -> None:
        """
        Record a detection event for *name*.

        - Creates a new `persons` row on first sighting.
        - Increments `detection_count` and updates `last_seen` on subsequent calls.
        - Always appends a row to `detection_events`.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._conn:
            # Upsert into persons ------------------------------------------------
            existing = self._conn.execute(
                "SELECT id FROM persons WHERE name = ?", (name,)
            ).fetchone()

            if existing is None:
                # First time we see this person
                self._conn.execute(
                    """
                    INSERT INTO persons (name, first_seen, last_seen, detection_count)
                    VALUES (?, ?, ?, 1)
                    """,
                    (name, now, now),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE persons
                    SET last_seen       = ?,
                        detection_count = detection_count + 1
                    WHERE name = ?
                    """,
                    (now, name),
                )

            # Always insert a fine-grained event row ----------------------------
            self._conn.execute(
                """
                INSERT INTO detection_events (name, detected_at, mask_status, similarity_score)
                VALUES (?, ?, ?, ?)
                """,
                (name, now, mask_status, float(similarity_score)),
            )

    def get_person_summary(self, name: str) -> dict | None:
        """Return the summary row for *name*, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM persons WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_persons(self) -> list[dict]:
        """Return all rows from the persons summary table, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM persons ORDER BY last_seen DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_events_for_person(self, name: str, limit: int = 100) -> list[dict]:
        """Return the most recent detection events for *name*."""
        rows = self._conn.execute(
            """
            SELECT * FROM detection_events
            WHERE name = ?
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Return the *limit* most recent detection events across all persons."""
        rows = self._conn.execute(
            """
            SELECT * FROM detection_events
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_person(self, name: str) -> bool:
        """Delete a person and all their events from the log DB."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM persons WHERE name = ?", (name,)
            )
            self._conn.execute(
                "DELETE FROM detection_events WHERE name = ?", (name,)
            )
        return cur.rowcount > 0

    def clear_all(self) -> None:
        """Wipe all data from both tables."""
        with self._conn:
            self._conn.executescript(
                "DELETE FROM detection_events; DELETE FROM persons;"
            )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
