# Demo Video Script — Desk (2 Minutes)

## Scene 1: Opening (0:00-0:15)

**[Screen: Terminal with uvicorn running]**

**Voiceover:**
"This is Desk — a local-first AI chief of staff for solo entrepreneurs. It runs entirely on your laptop with zero cloud dependency."

**[Show system stats: memory, CPU]**

---

## Scene 2: Task Creation (0:15-0:45)

**[Screen: Web UI at localhost:8000]**

**User types:** "Create a task to call the supplier tomorrow about the bulk order for school supplies"

**[Show plan card appearing]**

**Voiceover:**
"Desk creates a plan before executing. You review it, then confirm."

**[Show plan: create_task with title, due_date, tags]**

**User clicks:** "Execute Plan"

**[Show task created successfully]**

---

## Scene 3: Event with Conflict Detection (0:45-1:15)

**[Screen: Web UI]**

**User types:** "Schedule a meeting with the volunteer coordinator this Friday at 3pm"

**[Show plan: check_conflicts + create_event]**

**Voiceover:**
"Desk automatically checks for scheduling conflicts before creating events."

**[Show conflict check result: no conflicts]**

**User clicks:** "Execute Plan"

**[Show event created]**

---

## Scene 4: RAG Search (1:15-1:45)

**[Screen: Web UI]**

**User types:** "Search for notes about supplier communication preferences"

**[Show plan: search_notes with vector query]**

**Voiceover:**
"Desk uses vector search to find relevant notes — not just keywords."

**[Show results: notes about WhatsApp, phone, email preferences]**

---

## Scene 5: Closing (1:45-2:00)

**[Screen: All tasks, events, notes visible in UI]**

**Voiceover:**
"Desk: local-first, private, and built for entrepreneurs who need an AI assistant without the cloud. Built for ADTC 2026."

**[Show project structure, architecture diagram]**

---

## Recording Notes

- Use screen recording software (OBS, Camtasia, or built-in)
- Record in 1080p
- Add captions for accessibility
- Keep voiceover clear and concise
- Show real-time execution (don't speed up)
- Include system stats in corner (optional)