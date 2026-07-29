from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json


# --- Tool Parameter Schemas ---
class CreateTaskParams(BaseModel):
    title: str = Field(..., description="Task title")
    due_date: Optional[str] = Field(None, description="ISO format date, e.g., 2026-07-30 or 2026-07-30T14:00:00")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags")
    description: Optional[str] = Field(None, description="Optional description")

class UpdateTaskParams(BaseModel):
    task_id: int = Field(..., description="Task ID to update")
    title: Optional[str] = None
    due_date: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    status: Optional[str] = Field(None, description="pending, in_progress, done, cancelled")

class SearchTasksParams(BaseModel):
    query: str = Field(..., description="Search query for title, description, or tags")
    limit: int = Field(20, ge=1, le=100)

class CreateEventParams(BaseModel):
    title: str = Field(..., description="Event title")
    start_time: str = Field(..., description="ISO format start time, e.g., 2026-07-30T09:00:00")
    end_time: str = Field(..., description="ISO format end time, e.g., 2026-07-30T11:00:00")
    description: Optional[str] = None
    location: Optional[str] = None

class UpdateEventParams(BaseModel):
    event_id: int = Field(..., description="Event ID to update")
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None

class CheckConflictsParams(BaseModel):
    start_time: str = Field(..., description="ISO format start time")
    end_time: str = Field(..., description="ISO format end time")
    exclude_event_id: Optional[int] = None

class ListEventsParams(BaseModel):
    start: Optional[str] = Field(None, description="ISO format start of range")
    end: Optional[str] = Field(None, description="ISO format end of range")
    limit: int = Field(50, ge=1, le=200)

class CreateContactParams(BaseModel):
    name: str = Field(..., description="Contact name")
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None

class SearchContactsParams(BaseModel):
    query: str = Field(..., description="Search name, email, or company")
    limit: int = Field(20, ge=1, le=100)

class CreateNoteParams(BaseModel):
    content: str = Field(..., description="Note content")
    source: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)

class SearchNotesParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=50)

class DraftEmailParams(BaseModel):
    recipient: str = Field(..., description="Recipient email or name")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    tone: Optional[str] = Field("professional", description="professional, casual, urgent, friendly")
    context: Optional[str] = Field(None, description="Additional context for drafting")

class SummarizeDayParams(BaseModel):
    date: str = Field(..., description="ISO format date, e.g., 2026-07-30")


# --- Tool Registry ---
TOOLS = {
    "create_task": {
        "description": "Create a new task with title, optional due date, tags, and description",
        "params": CreateTaskParams,
        "returns": {"task_id": int, "title": str, "status": str}
    },
    "update_task": {
        "description": "Update an existing task by ID",
        "params": UpdateTaskParams,
        "returns": {"success": bool, "task_id": int}
    },
    "search_tasks": {
        "description": "Search tasks by keyword in title, description, or tags",
        "params": SearchTasksParams,
        "returns": {"tasks": List[Dict]}
    },
    "create_event": {
        "description": "Create a calendar event with conflict detection",
        "params": CreateEventParams,
        "returns": {"event_id": int, "title": str, "start_time": str, "end_time": str}
    },
    "update_event": {
        "description": "Update an existing calendar event",
        "params": UpdateEventParams,
        "returns": {"success": bool, "event_id": int}
    },
    "check_conflicts": {
        "description": "Check for calendar conflicts in a time range",
        "params": CheckConflictsParams,
        "returns": {"conflicts": List[Dict], "has_conflicts": bool}
    },
    "list_events": {
        "description": "List calendar events in a date range",
        "params": ListEventsParams,
        "returns": {"events": List[Dict]}
    },
    "create_contact": {
        "description": "Create a new contact",
        "params": CreateContactParams,
        "returns": {"contact_id": int, "name": str}
    },
    "search_contacts": {
        "description": "Search contacts by name, email, or company",
        "params": SearchContactsParams,
        "returns": {"contacts": List[Dict]}
    },
    "create_note": {
        "description": "Save a note with optional tags",
        "params": CreateNoteParams,
        "returns": {"note_id": int}
    },
    "search_notes": {
        "description": "Search notes by keyword",
        "params": SearchNotesParams,
        "returns": {"notes": List[Dict]}
    },
    "draft_email": {
        "description": "Draft an email (saved locally, not sent)",
        "params": DraftEmailParams,
        "returns": {"draft_id": int, "recipient": str, "subject": str}
    },
    "summarize_day": {
        "description": "Get a summary of tasks and events for a specific day",
        "params": SummarizeDayParams,
        "returns": {"date": str, "tasks": List[Dict], "events": List[Dict], "task_count": int, "event_count": int}
    },
}


# --- Tool Call Models ---
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None

class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: Dict[str, Any]
    error: Optional[str] = None


# --- Plan Model ---
class Plan(BaseModel):
    """A plan is a sequence of tool calls to execute"""
    reasoning: str = Field(..., description="Why this plan was chosen")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    requires_confirmation: bool = Field(True, description="Whether user must approve before execution")


# --- System Prompt Generator ---
def build_system_prompt() -> str:
    """Ultra-compact system prompt for minimum tokens, maximum accuracy."""
    return """Desk AI. JSON only. Tools:
-create_task(title*,due_date,tags:list)
-update_task(task_id*:int,title,due_date,status)
-search_tasks(query*,limit:int)
-create_event(title*,start_time*,end_time*,location)
-update_event(event_id*:int,title,start_time,end_time)
-check_conflicts(start_time*,end_time*)
-list_events(start,end,limit:int)
-create_contact(name*,email,phone,company)
-search_contacts(query*,limit:int)
-create_note(content*,tags:list)
-search_notes(query*,limit:int)
-draft_email(recipient*,subject*,body*:max100words,tone)
-summarize_day(date*)

Format: {"reasoning":"...","tool_calls":[{"id":"1","name":"tool","arguments":{}}]}
*required. ISO dates. Tags: ["a","b"].

Examples:
User: Task call supplier tomorrow
{"reasoning":"Create task","tool_calls":[{"id":"1","name":"create_task","arguments":{"title":"Call supplier","due_date":"2026-07-30","tags":["supplier"]}}]}

User: Meeting Tuesday 3pm
{"reasoning":"Create event","tool_calls":[{"id":"1","name":"create_event","arguments":{"title":"Meeting","start_time":"2026-08-05T15:00:00","end_time":"2026-08-05T16:00:00"}}]}

User: Notes supplier ABC then email
{"reasoning":"Search+draft","tool_calls":[{"id":"1","name":"search_notes","arguments":{"query":"supplier ABC"}},{"id":"2","name":"draft_email","arguments":{"recipient":"supplier","subject":"Follow-up","body":"Hi, following up..."}}]}"""


# --- Validation ---
def validate_tool_call(name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate a tool call against its schema. Returns (valid, error_message)."""
    if name not in TOOLS:
        return False, f"Unknown tool: {name}"
    try:
        TOOLS[name]["params"](**arguments)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_plan(plan: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate an entire plan. Returns (valid, error_message)."""
    if "tool_calls" not in plan:
        return False, "Plan missing 'tool_calls' field"
    
    # Normalize tool calls: handle "tool" key as alias for "name"
    for tc in plan["tool_calls"]:
        if "name" not in tc and "tool" in tc:
            tc["name"] = tc["tool"]
    
    # Filter out invalid tool calls
    valid_calls = []
    for tc in plan["tool_calls"]:
        if "name" not in tc:
            continue
        valid, err = validate_tool_call(tc["name"], tc.get("arguments", {}))
        if valid:
            valid_calls.append(tc)
    
    if not valid_calls:
        return False, "No valid tool calls in plan"
    
    plan["tool_calls"] = valid_calls
    return True, None


if __name__ == "__main__":
    # Print system prompt for reference
    print(build_system_prompt())


def build_tools_schema() -> List[Dict[str, Any]]:
    """Convert TOOLS registry to OpenAI function calling format for llama-cpp-python."""
    tools = []
    for name, info in TOOLS.items():
        params = info["params"].model_json_schema()
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": params,
            }
        })
    return tools