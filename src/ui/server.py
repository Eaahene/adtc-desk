"""
FastAPI web server for the local UI.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json

from src.orchestrator.orchestrator import Orchestrator
from src.models.llm import LocalLLM
from src.db.database import init_db

app = FastAPI(title="Desk - Local AI Chief of Staff")

# Initialize components on startup
llm = None
orchestrator = None
pending_plan = None


@app.on_event("startup")
async def startup():
    global llm, orchestrator
    init_db()
    llm = LocalLLM()
    orchestrator = Orchestrator(llm)


class ChatRequest(BaseModel):
    message: str
    auto_confirm: bool = False


class ConfirmRequest(BaseModel):
    plan: Dict[str, Any]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process user message and return plan or results."""
    global pending_plan
    result = orchestrator.run(request.message, auto_confirm=request.auto_confirm)
    
    if result["status"] == "awaiting_confirmation":
        pending_plan = result["plan"]
        return {
            "status": "awaiting_confirmation",
            "plan": result["plan"],
            "message": "Plan generated. Confirm to execute?"
        }
    elif result["status"] == "completed":
        pending_plan = None
        return {
            "status": "completed",
            "plan": result["plan"],
            "results": result["results"]
        }
    else:
        pending_plan = None
        return {
            "status": "error",
            "message": result.get("message", "Unknown error"),
            "raw": result.get("raw")
        }


@app.post("/api/confirm")
async def confirm(request: ConfirmRequest):
    """Execute a confirmed plan."""
    global pending_plan, orchestrator
    result = orchestrator.confirm_and_execute(request.plan)
    pending_plan = None
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve static files
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("src/ui/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)