
from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "relation_radar.db"


def get_db_path() -> Path:
    """
    Return the SQLite database path.

    For now this is derived from the project root. Later we can
    read from config/settings.toml if needed.
    """
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Get a SQLite connection with row_factory set to sqlite3.Row.
    Ensures the data directory exists.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the database schema (idempotent).

    Tables:
    - persons
    - events
    - event_persons (many-to-many between events and persons)
    - relationships
    - feedback (user ratings for QA answers)
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                nickname TEXT,
                birthday TEXT,
                gender TEXT,
                tags TEXT,        -- JSON-encoded list of strings
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT,     -- ISO datetime string
                raw_time_text TEXT,
                event_type TEXT,
                summary TEXT,
                raw_text TEXT,
                emotion TEXT,
                preferences TEXT,     -- JSON-encoded list of strings
                taboos TEXT,          -- JSON-encoded list of strings
                tags TEXT,            -- JSON-encoded list of strings
                embedding_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS event_persons (
                event_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                PRIMARY KEY (event_id, person_id),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_a_id INTEGER NOT NULL,
                person_b_id INTEGER NOT NULL,
                score REAL,
                relation_type TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_a_id) REFERENCES persons(id) ON DELETE CASCADE,
                FOREIGN KEY (person_b_id) REFERENCES persons(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);
            CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at);
            CREATE INDEX IF NOT EXISTS idx_event_persons_person ON event_persons(person_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_person_a ON relationships(person_a_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_person_b ON relationships(person_b_id);

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                used_context_ids TEXT, -- JSON-encoded list of event IDs
                rating TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_person ON feedback(person_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
            """
        )

        conn.commit()
    finally:
        conn.close()

