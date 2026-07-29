"""
SQLite database layer with sqlite-vec for vector search.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Try to import sqlite-vec, but handle gracefully if not available
try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

DB_PATH = Path(__file__).parent.parent.parent / "models" / "otimi.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """Get a database connection with sqlite-vec loaded if available."""
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    if HAS_SQLITE_VEC:
        try:
            sqlite_vec.load(conn)
        except Exception as e:
            print(f"Warning: Could not load sqlite-vec extension: {e}")
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_conn()
    conn.executescript("""
        -- Tasks table
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            due_date TEXT,
            tags TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Events/calendar table
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT,
            attendees TEXT,
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
            tags TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Notes table with vector embeddings
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            source_id INTEGER,
            tags TEXT,
            embedding BLOB,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Virtual table for vector search (sqlite-vec)
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
            embedding FLOAT[384]
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
        CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time);
        CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert sqlite3.Row to dict, parsing JSON fields."""
    d = dict(row)
    for key in ('tags', 'attendees'):
        if key in d and d[key]:
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d


# --- Task Operations ---
def create_task(title: str, description: str = "", due_date: str = "", tags: List[str] = None) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO tasks (title, description, due_date, tags, status) VALUES (?, ?, ?, ?, 'pending')",
        (title, description, due_date, json.dumps(tags or []))
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def update_task(task_id: int, **fields) -> bool:
    allowed = {'title', 'description', 'due_date', 'tags', 'status'}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if 'tags' in updates:
        updates['tags'] = json.dumps(updates['tags'])
    updates['updated_at'] = datetime.now().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = get_conn()
    cur = conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", (*updates.values(), task_id))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def delete_task(task_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def get_task(task_id: int) -> Optional[Dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return row_to_dict(row) if row else None


def search_tasks(query: str = "", status: str = "", limit: int = 20) -> List[Dict]:
    conn = get_conn()
    sql = "SELECT * FROM tasks WHERE 1=1"
    params = []
    if query:
        sql += " AND (title LIKE ? OR description LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# --- Event Operations ---
def create_event(title: str, start_time: str, end_time: str, description: str = "", location: str = "", attendees: List[str] = None) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO events (title, description, start_time, end_time, location, attendees) VALUES (?, ?, ?, ?, ?, ?)",
        (title, description, start_time, end_time, location, json.dumps(attendees or []))
    )
    conn.commit()
    event_id = cur.lastrowid
    conn.close()
    return event_id


def update_event(event_id: int, **fields) -> bool:
    allowed = {'title', 'description', 'start_time', 'end_time', 'location', 'attendees'}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if 'attendees' in updates:
        updates['attendees'] = json.dumps(updates['attendees'])
    updates['updated_at'] = datetime.now().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = get_conn()
    cur = conn.execute(f"UPDATE events SET {set_clause} WHERE id = ?", (*updates.values(), event_id))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def delete_event(event_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def get_event(event_id: int) -> Optional[Dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return row_to_dict(row) if row else None


def search_events(query: str = "", start_after: str = "", start_before: str = "", limit: int = 20) -> List[Dict]:
    conn = get_conn()
    sql = "SELECT * FROM events WHERE 1=1"
    params = []
    if query:
        sql += " AND (title LIKE ? OR description LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])
    if start_after:
        sql += " AND start_time >= ?"
        params.append(start_after)
    if start_before:
        sql += " AND start_time <= ?"
        params.append(start_before)
    sql += " ORDER BY start_time ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


def check_conflicts(start_time: str, end_time: str, exclude_id: int = None) -> List[Dict]:
    """Find events that overlap with the given time range."""
    conn = get_conn()
    sql = """SELECT * FROM events 
             WHERE start_time < ? AND end_time > ?"""
    params = [end_time, start_time]
    if exclude_id:
        sql += " AND id != ?"
        params.append(exclude_id)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# --- Contact Operations ---
def create_contact(name: str, email: str = "", phone: str = "", company: str = "", role: str = "", tags: List[str] = None, notes: str = "") -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO contacts (name, email, phone, company, role, tags, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, email, phone, company, role, json.dumps(tags or []), notes)
    )
    conn.commit()
    contact_id = cur.lastrowid
    conn.close()
    return contact_id


def search_contacts(query: str = "", limit: int = 20) -> List[Dict]:
    conn = get_conn()
    sql = "SELECT * FROM contacts WHERE 1=1"
    params = []
    if query:
        sql += " AND (name LIKE ? OR email LIKE ? OR company LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
    sql += " ORDER BY name ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# --- Note Operations (with vector search) ---
def create_note(content: str, source: str = "", source_id: int = None, tags: List[str] = None, embedding: List[float] = None) -> int:
    """Create a note with optional auto-generated embedding for RAG."""
    # Auto-generate embedding if not provided
    if embedding is None:
        try:
            from src.models.embeddings import embed_text
            embedding = embed_text(content)
        except Exception:
            pass  # Keyword search fallback if embedding fails
    
    conn = get_conn()
    embedding_blob = None
    if embedding and HAS_SQLITE_VEC:
        import sqlite_vec
        embedding_blob = sqlite_vec.serialize_float32(embedding)
    
    cur = conn.execute(
        "INSERT INTO notes (content, source, source_id, tags, embedding) VALUES (?, ?, ?, ?, ?)",
        (content, source, source_id, json.dumps(tags or []), embedding_blob)
    )
    note_id = cur.lastrowid
    
    if embedding and HAS_SQLITE_VEC:
        import sqlite_vec
        conn.execute("INSERT INTO notes_vec (rowid, embedding) VALUES (?, ?)", 
                     (note_id, sqlite_vec.serialize_float32(embedding)))
    
    conn.commit()
    conn.close()
    return note_id


def search_notes_by_vector(embedding: List[float], limit: int = 10) -> List[Dict]:
    """Vector similarity search using sqlite-vec."""
    if not HAS_SQLITE_VEC:
        return []
    
    conn = get_conn()
    import sqlite_vec
    rows = conn.execute(
        """SELECT n.*, distance FROM notes n
           JOIN notes_vec v ON n.id = v.rowid
           WHERE v.embedding MATCH ?
           AND k = ?""",
        (sqlite_vec.serialize_float32(embedding), limit)
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


def search_notes_keyword(query: str, limit: int = 20) -> List[Dict]:
    """Search notes using vector search (preferred) or keyword fallback."""
    # Try vector search first
    try:
        from src.models.embeddings import embed_text
        query_embedding = embed_text(query)
        results = search_notes_by_vector(query_embedding, limit)
        if results:
            return results
    except Exception:
        pass
    
    # Fallback to keyword search
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM notes WHERE content LIKE ? OR tags LIKE ? ORDER BY created_at DESC LIMIT ?",
        (f"%{query}%", f"%{query}%", limit)
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# --- Email Draft Operations ---
def create_email_draft(recipient: str, subject: str, body: str, tone: str = "professional") -> int:
    """Save an email draft (not sent). Stored as a note with source='email_draft'."""
    content = f"To: {recipient}\nSubject: {subject}\n\n{body}"
    return create_note(content, source="email_draft", tags=[tone], embedding=None)


# --- Summary ---
def summarize_day(date: str) -> Dict:
    """Get tasks and events for a specific date (YYYY-MM-DD)."""
    start = f"{date}T00:00:00"
    end = f"{date}T23:59:59"
    tasks = search_tasks()
    day_tasks = [t for t in tasks if t.get('due_date', '').startswith(date)]
    events = search_events(start_after=start, start_before=end)
    return {
        "date": date,
        "tasks": day_tasks,
        "events": events,
        "task_count": len(day_tasks),
        "event_count": len(events)
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")