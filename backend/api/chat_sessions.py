from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import get_current_user_id
from backend.database.db import ChatSession, get_db

router = APIRouter(prefix="/meal-planner", tags=["Meal Planner Sessions"])


class SaveSessionRequest(BaseModel):
    session_id: str | None = None
    title: str
    preview: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/sessions")
def save_session(
    req: SaveSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    session_id = req.session_id or str(uuid.uuid4())

    existing = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )

    if existing:
        existing.title = req.title
        existing.preview = req.preview
        existing.messages = json.dumps(req.messages)
        existing.updated_at = datetime.utcnow()
    else:
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=req.title,
            preview=req.preview,
            messages=json.dumps(req.messages),
        )
        db.add(session)

    db.commit()
    return {"id": session_id, "saved": True}


@router.get("/sessions")
def get_sessions(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
) -> list[dict[str, Any]]:
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(20)
        .all()
    )

    return [
        {
            "id": s.id,
            "title": s.title,
            "preview": s.preview,
            "messages": json.loads(s.messages),
            "created_at": s.created_at.isoformat() if s.created_at else "",
            "updated_at": s.updated_at.isoformat() if s.updated_at else "",
        }
        for s in sessions
    ]


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, bool]:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()
    return {"deleted": True}
