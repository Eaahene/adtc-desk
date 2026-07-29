"""
SQLite schema for tasks, events, contacts, and vector search.
Run once at startup to initialize the database.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "models" / "otimi.db"

SCHEMA = """
-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    due_date TEXT,           -- ISO 8601 datetime string
    tags TEXT,               -- JSON array of tags
    status TEXT DEFAULT 'pending',  -- pending, in_progress, done, cancelled
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Events/calendar table
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    start_time TEXT NOT NULL,  -- ISO 8601
    end_time TEXT NOT NULL,    -- ISO 8601
    location TEXT,
    attendees TEXT,            -- JSON array
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Contacts table
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    company TEXT,
    role TEXT,
    tags TEXT,                 -- JSON array
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Notes/memory table for RAG
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source TEXT,               -- 'task', 'event', 'contact', 'manual'
    source_id INTEGER,
    tags TEXT,                 -- JSON array
    embedding BLOB,            -- sqlite-vec vector
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_tags ON tasks(tags);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time);
CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);
CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at);

-- Vector index for notes (requires sqlite-vec extension)
-- CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
--     embedding FLOAT[384]
-- );
"""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    
    # Load sqlite-vec extension
    try:
        conn.load_extension("vec0")
        print("sqlite-vec extension loaded")
    except Exception as e:
        print(f"Warning: sqlite-vec not available: {e}")

    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()