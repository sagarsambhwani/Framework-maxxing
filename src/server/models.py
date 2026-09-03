"""Pydantic Request & Response Schemas for Server Endpoints."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text")
    model: str = Field(default="groq/qwen/qwen3.8-27b", description="Model engine slug")
    session_id: str = Field(default="default-session", description="Session identifier")
    guardrails_enabled: bool = Field(default=True, description="Toggle NeMo guardrails")
    history: List[Dict[str, str]] = Field(default=[], description="Recent conversation turns")


class TranscribeResponse(BaseModel):
    text: str
    duration_s: float
