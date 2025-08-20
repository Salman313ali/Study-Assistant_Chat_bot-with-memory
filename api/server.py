from __future__ import annotations

import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chatbot import StudyAssistant
from app.config import get_settings


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    style: Optional[str] = "short"  # "short" | "detailed"


class ChatResponse(BaseModel):
    answer: str
    key_points: list[str]
    suggested_questions: list[str]
    references: list[str]


class ResetResponse(BaseModel):
    ok: bool


def build_assistant_singleton() -> StudyAssistant:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return StudyAssistant(
        groq_api_key=settings.groq_api_key,
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        embeddings_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        persist_path=os.getenv("PERSIST_PATH", "./storage/memory_db"),
    )


assistant = build_assistant_singleton()

app = FastAPI(title="Study Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        print(f"Received style: {req.style}")  # Debug log
        result = assistant.ask(
            req.message,
            session_id=req.session_id or "default",
            style=(req.style or "short"),
        )
        return ChatResponse(
            answer=result.answer,
            key_points=result.key_points,
            suggested_questions=result.suggested_questions,
            references=result.references,
        )
    except Exception as exc:  # pragma: no cover - surface errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/session/{session_id}", response_model=ResetResponse)
def reset_session(session_id: str) -> ResetResponse:
    try:
        assistant.reset_session(session_id)
        return ResetResponse(ok=True)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Run with: uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

