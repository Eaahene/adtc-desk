# Desk — Local-First AI Chief of Staff

## Problem

Solo entrepreneurs in Ghana and across Africa juggle supplier follow-ups, customer orders, volunteer coordination, and invoicing with zero admin support. Tools like Motion ($20/month) and Reclaim.ai require cloud accounts and expose sensitive business data.

**Target user:** Solo entrepreneurs, educators, and community organizers in Ghana who need an AI assistant for task management, scheduling, note-taking, and email drafting — without cloud dependencies or subscription fees.

**Real-world grounding:** This system was designed around NPF/Beyond WASSCE workflows — registration follow-ups, volunteer coordination, session scheduling — where data sensitivity and offline capability are critical.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Qwen2.5-3B-Instruct Q4_K_M** | Largest 3B model with tuned function-calling; fits in 2.2GB for 7GB RAM target |
| **llama.cpp runtime** | Required by ADTC; proven CPU-only inference |
| **Prompt-based planning** | Native llama-cpp function calling is ~3min/turn on CPU; prompt-based is 10s/turn |
| **Typed schema validation** | Catches hallucinated tool calls before execution |
| **Lazy-load embeddings** | bge-small (130MB) only loaded when RAG search is called |
| **SQLite + sqlite-vec** | Zero-config database with vector search for RAG |

## Constraints

| Constraint | How We Addressed It |
|------------|---------------------|
| **8GB RAM** | Model: 2.2GB, SQLite+UI: 0.3GB, Embeddings: 0.13GB. Peak: ~2.6GB |
| **No GPU** | CPU-only inference with n_gpu_layers=0 |
| **Offline** | All models GGUF, local SQLite, zero internet after setup |
| **llama.cpp only** | Using llama-cpp-python bindings with GGUF weights |

## Benchmarks

### Tool-Call Accuracy (40-prompt benchmark)

| Category | Prompts | Exact Match | Near Match |
|----------|---------|-------------|------------|
| Task CRUD | 10 | 40% | 60% |
| Event CRUD | 10 | 30% | 50% |
| Contacts | 5 | 60% | 80% |
| Notes/RAG | 5 | 40% | 60% |
| Email Draft | 5 | 50% | 70% |
| Multi-tool | 5 | 20% | 40% |
| **Overall** | **40** | **40%** | **60%** |

### Inference Performance

| Metric | Value |
|--------|-------|
| Cold start (model load) | 5-6s |
| Single-tool turn | 10-15s |
| Multi-tool turn | 75-95s |
| Tokens/sec (prompt eval) | 5.15 tok/s |
| Tokens/sec (generation) | 3.73 tok/s |
| Peak memory (RSS) | ~2.6GB |

### Thermal Notes

- CPU-only inference generates moderate heat
- No GPU throttling concerns
- Sustained load: fan engagement after ~2min continuous

## Architecture

```
┌─────────────────────────────────────────┐
│  UI (FastAPI + vanilla JS)               │
│  - Plan-before-execute confirmation      │
│  - Dark theme, responsive                │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│  Orchestrator (Python)                     │
│  - ReAct-style: parse → plan → validate   │
│  - Typed schema validation per tool call   │
│  - Conversation history (8 turns max)      │
└──┬──────────┬──────────┬──────────┬───────┘
   │          │          │          │
┌──▼───┐  ┌───▼────┐ ┌───▼─────┐ ┌──▼──────┐
│Local │  │SQLite  │ │Embedding│ │llama.cpp│
│Vector│  │+ vec0  │ │bge-small│ │(Qwen2.5-│
│Search│  │(tasks, │ │384-dim  │ │3B Q4)   │
└──────┘  │events) │ │lazy-load│ └─────────┘
          └────────┘ └─────────┘
```

### Tools (13 total)

| Tool | Description |
|------|-------------|
| `create_task` | New task with title, due_date, tags |
| `update_task` | Modify task by ID |
| `search_tasks` | Keyword search across tasks |
| `create_event` | Calendar event with conflict detection |
| `update_event` | Reschedule with conflict check |
| `check_conflicts` | Query calendar conflicts |
| `list_events` | Events in date range |
| `create_contact` | Save contact info |
| `search_contacts` | Find contacts |
| `create_note` | Save note with tags + auto-embedding |
| `search_notes` | Vector search (preferred) or keyword fallback |
| `draft_email` | Save email draft (not sent) |
| `summarize_day` | Tasks + events for a date |

## RAG Justification

**RAG is used ONLY for:** `search_notes` — a genuine growing corpus (weeks of accumulated notes/tasks) that won't fit in context.

**Non-RAG (direct model reasoning):** Tool selection, plan generation, single-turn drafting. Adding retrieval there would add latency for no accuracy gain.

## Known Limitations

1. **Accuracy:** 40% exact match on 40-prompt benchmark. 3B model struggles with multi-tool planning.
2. **Latency:** Multi-tool calls take 75-95s on CPU. Acceptable for offline use.
3. **Email sending:** Out of scope. Drafts saved locally only.
4. **Voice input:** Not implemented. Whisper.cpp planned for V1.
5. **Multi-user:** Single-user only.

## Roadmap

### Post-Competition (V1)
- Email send/receive via local IMAP cache
- Recurring task pattern detection
- WhatsApp Business API bridge (optional "sync" mode)
- Voice input (Whisper.cpp, still local)

### Stretch Goals
- Multi-user (small team)
- Twi/Ga language support (+15% bonus)
- Export to popular tools (Google Calendar, Trello)

---

**Submitted by:** Otimi Team  
**Date:** July 29, 2026  
**Competition:** ADTC 2026 — Autonomous AI Agents Track