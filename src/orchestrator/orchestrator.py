"""
Orchestrator using prompt-based planning (faster than native function calling).
"""
import json
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from src.tools.registry import TOOLS, validate_tool_call, validate_plan, build_system_prompt
from src.models.llm import LocalLLM, parse_json_response
from src.db.database import (
    create_task, update_task, delete_task, search_tasks, get_task,
    create_event, update_event, delete_event, search_events, check_conflicts, get_event,
    create_contact, search_contacts,
    create_note, search_notes_keyword, search_notes_by_vector,
    create_email_draft, summarize_day,
)


# --- Tool Executors ---
def exec_create_task(args: Dict) -> Dict:
    task_id = create_task(
        title=args["title"],
        description=args.get("description", ""),
        due_date=args.get("due_date", ""),
        tags=args.get("tags", []),
    )
    return {"task_id": task_id, "title": args["title"], "status": "pending"}

def exec_update_task(args: Dict) -> Dict:
    task_id = args.pop("task_id")
    success = update_task(task_id, **args)
    return {"success": success, "task_id": task_id}

def exec_search_tasks(args: Dict) -> Dict:
    tasks = search_tasks(
        query=args.get("query", ""),
        limit=args.get("limit", 20),
    )
    return {"tasks": tasks}

def exec_create_event(args: Dict) -> Dict:
    conflicts = check_conflicts(args["start_time"], args["end_time"])
    if conflicts:
        return {
            "error": "Conflicts detected",
            "conflicts": conflicts,
            "has_conflicts": True,
        }
    event_id = create_event(
        title=args["title"],
        start_time=args["start_time"],
        end_time=args["end_time"],
        description=args.get("description", ""),
        location=args.get("location", ""),
    )
    return {"event_id": event_id, "title": args["title"], "start_time": args["start_time"], "end_time": args["end_time"]}

def exec_update_event(args: Dict) -> Dict:
    event_id = args.pop("event_id")
    if "start_time" in args or "end_time" in args:
        event = get_event(event_id)
        if event:
            start = args.get("start_time", event["start_time"])
            end = args.get("end_time", event["end_time"])
            conflicts = check_conflicts(start, end, exclude_id=event_id)
            if conflicts:
                return {"error": "Conflicts detected", "conflicts": conflicts, "has_conflicts": True}
    success = update_event(event_id, **args)
    return {"success": success, "event_id": event_id}

def exec_check_conflicts(args: Dict) -> Dict:
    conflicts = check_conflicts(
        args["start_time"],
        args["end_time"],
        exclude_id=args.get("exclude_event_id"),
    )
    return {"conflicts": conflicts, "has_conflicts": len(conflicts) > 0}

def exec_list_events(args: Dict) -> Dict:
    events = search_events(
        query=args.get("query", ""),
        start_after=args.get("start", ""),
        start_before=args.get("end", ""),
        limit=args.get("limit", 50),
    )
    return {"events": events}

def exec_create_contact(args: Dict) -> Dict:
    contact_id = create_contact(
        name=args["name"],
        email=args.get("email", ""),
        phone=args.get("phone", ""),
        company=args.get("company", ""),
        notes=args.get("notes", ""),
    )
    return {"contact_id": contact_id, "name": args["name"]}

def exec_search_contacts(args: Dict) -> Dict:
    contacts = search_contacts(query=args.get("query", ""), limit=args.get("limit", 20))
    return {"contacts": contacts}

def exec_create_note(args: Dict) -> Dict:
    note_id = create_note(
        content=args["content"],
        source=args.get("source", ""),
        tags=args.get("tags", []),
        embedding=None,
    )
    return {"note_id": note_id}

def exec_search_notes(args: Dict) -> Dict:
    notes = search_notes_keyword(query=args.get("query", ""), limit=args.get("limit", 10))
    return {"notes": notes}

def exec_draft_email(args: Dict) -> Dict:
    draft_id = create_email_draft(
        recipient=args["recipient"],
        subject=args["subject"],
        body=args["body"],
        tone=args.get("tone", "professional"),
    )
    return {"draft_id": draft_id, "recipient": args["recipient"], "subject": args["subject"]}

def exec_summarize_day(args: Dict) -> Dict:
    return summarize_day(args["date"])


# --- Tool Executor Map ---
EXECUTORS: Dict[str, Callable] = {
    "create_task": exec_create_task,
    "update_task": exec_update_task,
    "search_tasks": exec_search_tasks,
    "create_event": exec_create_event,
    "update_event": exec_update_event,
    "check_conflicts": exec_check_conflicts,
    "list_events": exec_list_events,
    "create_contact": exec_create_contact,
    "search_contacts": exec_search_contacts,
    "create_note": exec_create_note,
    "search_notes": exec_search_notes,
    "draft_email": exec_draft_email,
    "summarize_day": exec_summarize_day,
}


@dataclass
class Orchestrator:
    llm: LocalLLM
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = field(default_factory=build_system_prompt)
    
    def _build_prompt(self, user_input: str) -> str:
        """Build Qwen2.5 chat format prompt with system prompt and history."""
        parts = [f"<|system|>\n{self.system_prompt}\n"]
        for msg in self.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append(f"<|user|>\n{user_input}\n")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
    
    def plan(self, user_input: str) -> Dict[str, Any]:
        """Generate a plan from user input using generate()."""
        prompt = self._build_prompt(user_input)
        
        result = self.llm.generate(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["<|user|>", "<|system|>"],
        )
        
        if isinstance(result, str):
            text = result
        else:
            text = result.get("choices", [{}])[0].get("text", "")
        
        plan = parse_json_response(text)
        if not plan:
            return {"error": "Failed to parse plan", "raw_response": text}
        
        return plan
    
    def validate(self, plan: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate plan against tool schemas."""
        return validate_plan(plan)
    
    def execute(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all tool calls in plan sequentially."""
        results = []
        for tool_call in plan.get("tool_calls", []):
            name = tool_call["name"]
            args = tool_call["arguments"]
            call_id = tool_call.get("id", "1")
            
            if name not in EXECUTORS:
                results.append({
                    "tool_call_id": call_id,
                    "name": name,
                    "error": f"Unknown tool: {name}",
                })
                continue
            
            try:
                result = EXECUTORS[name](args)
                results.append({
                    "tool_call_id": call_id,
                    "name": name,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "tool_call_id": call_id,
                    "name": name,
                    "error": str(e),
                })
        
        return results
    
    def run(self, user_input: str, auto_confirm: bool = False) -> Dict[str, Any]:
        """Full orchestration loop: plan -> validate -> confirm -> execute."""
        plan = self.plan(user_input)
        
        if "error" in plan:
            return {"status": "error", "message": plan["error"], "raw": plan.get("raw_response")}
        
        valid, error = self.validate(plan)
        if not valid:
            return {"status": "error", "message": f"Plan validation failed: {error}", "plan": plan}
        
        if plan.get("requires_confirmation", True) and not auto_confirm:
            return {
                "status": "awaiting_confirmation",
                "plan": plan,
                "message": "Plan generated. Confirm to execute?",
            }
        
        results = self.execute(plan)
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": json.dumps({"plan": plan, "results": results})})
        if len(self.conversation_history) > 16:
            self.conversation_history = self.conversation_history[-16:]
        
        return {
            "status": "completed",
            "plan": plan,
            "results": results,
        }
    
    def confirm_and_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a previously confirmed plan."""
        valid, error = self.validate(plan)
        if not valid:
            return {"status": "error", "message": f"Plan validation failed: {error}", "plan": plan}
        
        results = self.execute(plan)
        return {"status": "completed", "plan": plan, "results": results}


def create_orchestrator(model_path: str = None) -> Orchestrator:
    """Factory function to create orchestrator with loaded model."""
    llm = LocalLLM()
    return Orchestrator(llm=llm)


if __name__ == "__main__":
    llm = LocalLLM()
    orch = Orchestrator(llm)
    result = orch.run("Create a task to follow up with supplier ABC by Friday", auto_confirm=True)
    print(json.dumps(result, indent=2))