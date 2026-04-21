from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models.database import get_db
from models.db import Task
from routers.auth import require_auth
from services.task_access import build_task_query, get_task_for_user

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskCreate(BaseModel):
    name: str


class TaskSummaryResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str


class TaskDetailResponse(TaskSummaryResponse):
    negotiation_summary: dict | None = None
    offline_evaluation: dict | None = None
    training_plan: dict | None = None
    delivery_package: dict | None = None


def _task_summary_response(task: Task) -> TaskSummaryResponse:
    return TaskSummaryResponse(
        id=task.id,
        name=task.name,
        status=task.status,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )


def _task_detail_response(task: Task) -> TaskDetailResponse:
    return TaskDetailResponse(
        **_task_summary_response(task).model_dump(),
        negotiation_summary=task.negotiation_summary,
        offline_evaluation=task.offline_evaluation,
        training_plan=task.training_plan,
        delivery_package=task.delivery_package,
    )


@router.get("", response_model=list[TaskSummaryResponse])
def list_tasks(
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    tasks = build_task_query(db, current_user).order_by(Task.created_at.desc()).all()
    return [_task_summary_response(task) for task in tasks]


@router.get("/{task_id}", response_model=TaskDetailResponse)
def get_task(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    return _task_detail_response(task)


@router.post("", response_model=TaskSummaryResponse)
def create_task(
    payload: TaskCreate,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = Task(name=payload.name, owner_user_id=current_user["user_id"])
    db.add(task)
    db.commit()
    db.refresh(task)
    return _task_summary_response(task)


class TaskCloneRequest(BaseModel):
    new_name: str | None = None
    include_plan: bool = True
    include_train_config: bool = True
    include_augment_config: bool = True
    as_template: bool = False


@router.post("/{task_id}/clone", response_model=TaskSummaryResponse)
def clone_task(
    task_id: str,
    payload: TaskCloneRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    克隆一个任务。复制算法方案、训练配置、增强配置等到新任务。
    数据目录和训练产物不会被复制。
    """
    import copy
    source = get_task_for_user(db, task_id, current_user)

    suffix = "（模板）" if payload.as_template else "（克隆）"
    new_name = payload.new_name or f"{source.name} {suffix}"

    new_task = Task(name=new_name, owner_user_id=current_user["user_id"])

    if payload.include_plan and source.algorithm_plan:
        cloned_plan = copy.deepcopy(source.algorithm_plan)
        cloned_plan.pop("revision_snapshots", None)
        cloned_plan.pop("revision_history", None)
        new_task.algorithm_plan = cloned_plan
        new_task.algorithm_plan_status = "draft"
        new_task.pipeline_config = copy.deepcopy(source.pipeline_config) if source.pipeline_config else None
        new_task.negotiation_summary = copy.deepcopy(source.negotiation_summary) if source.negotiation_summary else None

    if payload.include_train_config and source.train_config:
        new_task.train_config = copy.deepcopy(source.train_config)

    if payload.include_augment_config and source.augment_config:
        new_task.augment_config = copy.deepcopy(source.augment_config)

    if payload.as_template:
        new_task.status = "template"

    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return _task_summary_response(new_task)


@router.delete("/{task_id}")
def delete_task(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    db.delete(task)
    db.commit()
    return {"code": 0, "msg": "ok", "data": None}
