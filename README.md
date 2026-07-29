# Desk — Local-First AI Chief of Staff

A lightweight, offline-first AI assistant for solo entrepreneurs. Desk helps manage tasks, calendar, notes, contacts, and email drafts through natural conversation — all running locally on your machine with zero cloud dependency.

**Built for ADTC 2026 — Local-First Workflow Automation**

---

## Features

- **13 tools:** task CRUD, event creation with conflict detection, contact management, note-taking with vector search, email drafting, daily summaries
- **Plan-before-execute:** AI creates a plan, you review and confirm before execution
- **RAG search:** Vector similarity search over notes using bge-small embeddings
- **Offline-first:** No internet required after initial setup
- **Privacy-first:** All data stays on your machine

## Architecture

```
┌─────────────────────────────────────────┐
│  UI (FastAPI + vanilla JS)               │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│  Orchestrator (plan → validate → execute)  │
└──┬──────────┬──────────┬──────────┬───────┘
   │          │          │          │
┌──▼───┐  ┌───▼────┐ ┌───▼─────┐ ┌──▼──────┐
│Local │  │SQLite  │ │Embedding│ │llama.cpp│
│Vector│  │+ vec0  │ │bge-small│ │(Qwen2.5-│
│Search│  │(tasks, │ │384-dim  │ │3B Q4)   │
└──────┘  │events) │ │lazy-load│ └─────────┘
          └────────┘ └─────────┘
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/desk.git
cd desk
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 3. Download models (~2.5GB)
python scripts/download_models.py

# 4. Start the server
python -m src.cli
# or
uvicorn src.ui.server:app --reload

# 5. Open browser
# http://localhost:8000
```

## Benchmark Results

| Metric | Value |
|--------|-------|
| Tool-call accuracy | 40% exact, 60% near-match |
| Single-tool turn | 10-15s |
| Multi-tool turn | 75-95s |
| Peak memory | ~2.6GB |
| Model size | ~2.2GB |

## Project Structure

```
desk/
├── src/
│   ├── orchestrator/    # Core plan→validate→execute loop
│   ├── tools/           # 13 tool schemas + executors
│   ├── models/          # LLM wrapper + embeddings
│   ├── db/              # SQLite + sqlite-vec
│   └── ui/              # FastAPI server + web UI
├── eval/                # Benchmark suite (40 prompts)
├── scripts/             # Model download utilities
├── models/              # GGUF models (gitignored)
├── REPORT.md            # Competition report
└── requirements.txt
```

## Tools

| Tool | Description |
|------|-------------|
| `create_task` | Create a new task |
| `update_task` | Update existing task |
| `search_tasks` | Search tasks by keyword |
| `create_event` | Create calendar event (conflict check) |
| `update_event` | Reschedule event |
| `check_conflicts` | Check for scheduling conflicts |
| `list_events` | List events in date range |
| `create_contact` | Save contact |
| `search_contacts` | Find contacts |
| `create_note` | Save note (auto-embedded) |
| `search_notes` | Vector search notes |
| `draft_email` | Draft email (local only) |
| `summarize_day` | Daily summary |

## Tech Stack

- **LLM:** Qwen2.5-3B-Instruct (GGUF Q4_K_M)
- **Embeddings:** bge-small-en-v1.5 (GGUF F16)
- **Runtime:** llama-cpp-python (CPU-only)
- **Database:** SQLite + sqlite-vec
- **Backend:** FastAPI
- **Frontend:** Vanilla JS (dark theme)

## License

MIT

## Acknowledgments

- Built for ADTC 2026 competition
- Powered by llama.cpp, sqlite-vec, and HuggingFace models