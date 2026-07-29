#!/usr/bin/env python3
"""
CLI entry point for the Desk/Otimi agent.
Usage: python -m src.cli [command]
"""
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.orchestrator import create_orchestrator


def main():
    parser = argparse.ArgumentParser(description="Desk - Local AI Chief of Staff")
    parser.add_argument("--auto", action="store_true", help="Auto-confirm all plans")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("input", nargs="*", help="User input (if omitted, enters interactive mode)")
    args = parser.parse_args()
    
    print("Starting Desk...")
    orchestrator = create_orchestrator(model_path=args.model)
    
    if args.input:
        # Single command mode
        user_input = " ".join(args.input)
        result = orchestrator.run(user_input, auto_confirm=args.auto)
        print(json.dumps(result, indent=2))
        if result["status"] == "awaiting_confirmation":
            confirm = input("Execute? (y/n): ").strip().lower()
            if confirm == "y":
                result = orchestrator.confirm_and_execute(result["plan"])
                print(json.dumps(result, indent=2))
    else:
        # Interactive mode
        print("Desk interactive mode. Type 'quit' to exit.")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                
                result = orchestrator.run(user_input, auto_confirm=args.auto)
                print(f"\nStatus: {result['status']}")
                
                if result['status'] == 'awaiting_confirmation':
                    plan = result['plan']
                    print(f"Reasoning: {plan.get('reasoning', 'N/A')}")
                    print("Tool calls:")
                    for tc in plan.get('tool_calls', []):
                        print(f"  - {tc['name']}({json.dumps(tc['arguments'])})")
                    
                    confirm = input("\nExecute? (y/n/edit): ").strip().lower()
                    if confirm == "y":
                        result = orchestrator.confirm_and_execute(plan)
                        print_results(result['results'])
                    elif confirm == "edit":
                        print("Edit not implemented yet. Skipping.")
                
                elif result['status'] == 'completed':
                    print_results(result['results'])
                
                else:
                    print(f"Error: {result.get('message')}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def print_results(results: list):
    for r in results:
        if "error" in r:
            print(f"  ✗ {r['name']}: {r['error']}")
        else:
            print(f"  ✓ {r['name']}: {json.dumps(r.get('result', {}))}")


if __name__ == "__main__":
    main()