from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Query, Session

from models.db import Task


def build_task_query(db: Session, current_user: dict) -> Query:
    query = db.query(Task)
    if current_user.get("role") == "admin":
        return query.filter(or_(Task.owner_user_id == current_user["user_id"], Task.owner_user_id.is_(None)))
    return query.filter(Task.owner_user_id == current_user["user_id"])


def get_task_for_user(db: Session, task_id: str, current_user: dict) -> Task:
    task = build_task_query(db, current_user).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
