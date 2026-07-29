"""
Evaluation script for the 40-prompt tool-call accuracy benchmark.
Run: python eval/benchmark.py
"""
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from src.models.llm import LocalLLM, build_prompt, parse_json_response
from src.tools.registry import TOOLS, validate_plan
from src.orchestrator.orchestrator import Orchestrator


EVAL_PATH = Path(__file__).parent / "prompts.json"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"


@dataclass
class BenchmarkResult:
    prompt: str
    expected_tools: List[str]
    predicted_tools: List[str]
    exact_match: bool
    near_match: bool  # Right tools, minor arg differences
    latency_ms: float
    error: str = ""


def load_prompts() -> List[Dict]:
    """Load evaluation prompts from JSON."""
    if not EVAL_PATH.exists():
        # Create default prompts from NPF/Beyond WASSCE scenarios
        prompts = create_default_prompts()
        save_prompts(prompts)
        return prompts
    with open(EVAL_PATH) as f:
        return json.load(f)


def save_prompts(prompts: List[Dict]):
    with open(EVAL_PATH, "w") as f:
        json.dump(prompts, f, indent=2)


def create_default_prompts() -> List[Dict]:
    """Create 40 realistic prompts from NPF/Beyond WASSCE workflows."""
    return [
        # Task management (10)
        {"prompt": "Create a task to follow up with supplier ABC about the fabric delivery", "expected_tools": ["create_task"]},
        {"prompt": "Add a task: review volunteer applications for NPF program, due Friday", "expected_tools": ["create_task"]},
        {"prompt": "Create tasks for: email venue coordinator, confirm catering numbers, print badges", "expected_tools": ["create_task"]},
        {"prompt": "Mark the 'supplier follow-up' task as done", "expected_tools": ["update_task"]},
        {"prompt": "Update the venue booking task to change due date to Thursday", "expected_tools": ["update_task"]},
        {"prompt": "Search for tasks tagged 'supplier'", "expected_tools": ["search_tasks"]},
        {"prompt": "Find all pending tasks about volunteer coordination", "expected_tools": ["search_tasks"]},
        {"prompt": "Show me tasks due this week", "expected_tools": ["search_tasks"]},
        {"prompt": "Delete the old task about printing flyers", "expected_tools": ["delete_task"]},
        {"prompt": "Create a task to draft Monday's team standup notes", "expected_tools": ["create_task"]},
        
        # Calendar/Events (10)
        {"prompt": "Schedule NPF volunteer call Thursday 9-11am", "expected_tools": ["create_event"]},
        {"prompt": "Block Friday 2-4pm for Beyond WASSCE curriculum review", "expected_tools": ["create_event"]},
        {"prompt": "Check if I have conflicts on Wednesday 10am-12pm", "expected_tools": ["check_conflicts"]},
        {"prompt": "Create event: supplier meeting Tuesday 3pm-4pm at office", "expected_tools": ["create_event"]},
        {"prompt": "Move the Thursday volunteer call to 10am-12pm", "expected_tools": ["update_event"]},
        {"prompt": "Show my calendar events for next week", "expected_tools": ["list_events"]},
        {"prompt": "What do I have scheduled for Thursday?", "expected_tools": ["list_events"]},
        {"prompt": "Cancel the Friday curriculum review", "expected_tools": ["delete_event"]},
        {"prompt": "Check conflicts for Monday 9-11am before booking", "expected_tools": ["check_conflicts"]},
        {"prompt": "List all events between July 28 and August 3", "expected_tools": ["list_events"]},
        
        # Contacts (5)
        {"prompt": "Add contact: Kwame Asare, kwame@email.com, freelance designer", "expected_tools": ["create_contact"]},
        {"prompt": "Save supplier contact: Ama Mensah, ama@supplies.gh, fabric supplier", "expected_tools": ["create_contact"]},
        {"prompt": "Find contacts at 'TechHub' company", "expected_tools": ["search_contacts"]},
        {"prompt": "Search for contacts with email containing 'supplier'", "expected_tools": ["search_contacts"]},
        {"prompt": "Add volunteer coordinator: Grace Osei, grace@npf.org", "expected_tools": ["create_contact"]},
        
        # Notes/RAG (5)
        {"prompt": "Save note: Supplier ABC prefers WhatsApp for urgent updates", "expected_tools": ["create_note"]},
        {"prompt": "Note: Volunteer training sessions work best on weekends", "expected_tools": ["create_note"]},
        {"prompt": "Search notes about supplier communication preferences", "expected_tools": ["search_notes"]},
        {"prompt": "Find notes mentioning 'curriculum'", "expected_tools": ["search_notes"]},
        {"prompt": "Save note: Next NPF cohort starts Sept 15, need 20 volunteers", "expected_tools": ["create_note"]},
        
        # Email drafts (5)
        {"prompt": "Draft email to supplier ABC: follow up on fabric delivery, professional tone", "expected_tools": ["draft_email"]},
        {"prompt": "Write email draft to volunteers about Thursday call, friendly tone", "expected_tools": ["draft_email"]},
        {"prompt": "Compose email to venue: confirm booking for Aug 15, urgent tone", "expected_tools": ["draft_email"]},
        {"prompt": "Draft follow-up email to Kwame about design revisions, casual tone", "expected_tools": ["draft_email"]},
        {"prompt": "Email draft: thank sponsors for NPF support, professional", "expected_tools": ["draft_email"]},
        
        # Multi-tool / complex (5)
        {"prompt": "Find supplier tasks and draft follow-up emails for all 3", "expected_tools": ["search_tasks", "draft_email"]},
        {"prompt": "Check Thursday morning for conflicts, then block 9-11am for volunteer call", "expected_tools": ["check_conflicts", "create_event"]},
        {"prompt": "Search notes for 'volunteer', then create task to recruit 5 more", "expected_tools": ["search_notes", "create_task"]},
        {"prompt": "Find all contacts at 'NPF', draft thank you email to each", "expected_tools": ["search_contacts", "draft_email"]},
        {"prompt": "Summarize my day for Friday July 31", "expected_tools": ["summarize_day"]},
    ]


def run_benchmark() -> List[BenchmarkResult]:
    """Run the 40-prompt benchmark."""
    prompts = load_prompts()
    llm = LocalLLM()
    orchestrator = Orchestrator(llm)
    
    results = []
    
    print(f"Running benchmark on {len(prompts)} prompts...")
    print("=" * 60)
    
    for i, item in enumerate(prompts, 1):
        prompt = item["prompt"]
        expected = item["expected_tools"]
        
        print(f"\n[{i}/{len(prompts)}] {prompt[:60]}...")
        
        start = time.time()
        try:
            # Get plan without executing
            plan = orchestrator.plan(prompt)
            latency_ms = (time.time() - start) * 1000
            
            if "error" in plan:
                results.append(BenchmarkResult(
                    prompt=prompt,
                    expected_tools=expected,
                    predicted_tools=[],
                    exact_match=False,
                    near_match=False,
                    latency_ms=latency_ms,
                    error=plan["error"]
                ))
                print(f"  ✗ Error: {plan['error']}")
                continue
            
            predicted = [tc["name"] for tc in plan.get("tool_calls", [])]
            
            # Exact match: same tools in same order
            exact_match = predicted == expected
            
            # Near match: same tools (order doesn't matter)
            near_match = set(predicted) == set(expected)
            
            result = BenchmarkResult(
                prompt=prompt,
                expected_tools=expected,
                predicted_tools=predicted,
                exact_match=exact_match,
                near_match=near_match,
                latency_ms=latency_ms,
            )
            results.append(result)
            
            status = "✓" if exact_match else ("~" if near_match else "✗")
            print(f"  {status} Expected: {expected}")
            print(f"    Got:      {predicted} ({latency_ms:.0f}ms)")
            
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            results.append(BenchmarkResult(
                prompt=prompt,
                expected_tools=expected,
                predicted_tools=[],
                exact_match=False,
                near_match=False,
                latency_ms=latency_ms,
                error=str(e)
            ))
            print(f"  ✗ Exception: {e}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    total = len(results)
    exact = sum(1 for r in results if r.exact_match)
    near = sum(1 for r in results if r.near_match)
    errors = sum(1 for r in results if r.error)
    latencies = [r.latency_ms for r in results if not r.error]
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total prompts:     {total}")
    print(f"Exact match:       {exact} ({exact/total*100:.1f}%)")
    print(f"Near match:        {near} ({near/total*100:.1f}%)")
    print(f"Errors:            {errors} ({errors/total*100:.1f}%)")
    if latencies:
        print(f"Avg latency:       {statistics.mean(latencies):.0f}ms")
        print(f"Median latency:    {statistics.median(latencies):.0f}ms")
        print(f"P95 latency:       {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
    
    # Per-tool breakdown
    print("\nPer-tool accuracy:")
    tool_stats = {}
    for r in results:
        for tool in r.expected_tools:
            if tool not in tool_stats:
                tool_stats[tool] = {"total": 0, "correct": 0}
            tool_stats[tool]["total"] += 1
            if tool in r.predicted_tools:
                tool_stats[tool]["correct"] += 1
    
    for tool, stats in sorted(tool_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {tool:20s} {stats['correct']}/{stats['total']} ({acc:.1f}%)")


def save_results(results: List[BenchmarkResult]):
    """Save detailed results to JSON."""
    data = [asdict(r) for r in results]
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    results = run_benchmark()
    print_summary(results)
    save_results(results)