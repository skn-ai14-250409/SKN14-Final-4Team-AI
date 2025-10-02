from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.services.orchestrator import run_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])

class Profile(BaseModel):
    gender: Optional[str] = None

class ChatRequest(BaseModel):
    query: str = Field(..., description="자연어 질의")
    profile: Optional[Profile] = None

class ChatResponse(BaseModel):
    tool: str
    slots: Dict[str, Any]
    text: str
    ask: Optional[str] = None
    targets: Optional[List[str]] = None
    note: str
    meta: Optional[Dict[str, Any]] = None

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    payload = await run_pipeline(req.query, (req.profile or Profile()).model_dump())
    return ChatResponse(**payload)
