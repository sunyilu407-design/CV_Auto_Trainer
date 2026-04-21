"""模型缓存管理 API：列表 / 查看 / 删除已训练的模型缓存。"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from routers.auth import require_auth
from services.model_registry import get_model_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class CachedModelEntry(BaseModel):
    cache_id: str
    source_model_id: str
    task_id: str
    classes: List[str]
    class_count: int
    scenario_type: str
    map50: float | None = None
    map50_95: float | None = None
    weight_path: str
    trained_at: float
    image_count: int
    epochs_completed: int
    reuse_count: int
    tags: List[str] = []
    weight_exists: bool = False
    weight_size_mb: float | None = None


@router.get("/cache", response_model=List[CachedModelEntry])
def list_cached_models(current_user: dict = Depends(require_auth)):
    """列出所有已训练的模型缓存。"""
    registry = get_model_registry()
    entries = registry.list_cached_models()
    result: List[CachedModelEntry] = []
    for c in entries:
        raw: Dict[str, Any] = asdict(c)
        weight_path = raw.get("weight_path") or ""
        exists = bool(weight_path) and Path(weight_path).exists()
        size_mb: float | None = None
        if exists:
            try:
                size_mb = round(Path(weight_path).stat().st_size / (1024 * 1024), 2)
            except OSError:
                size_mb = None
        raw["weight_exists"] = exists
        raw["weight_size_mb"] = size_mb
        result.append(CachedModelEntry(**raw))
    # 按训练时间倒序
    result.sort(key=lambda e: e.trained_at or 0, reverse=True)
    return result


@router.delete("/cache/{cache_id}")
def delete_cached_model(
    cache_id: str,
    delete_weight_file: bool = True,
    current_user: dict = Depends(require_auth),
):
    """从注册表删除缓存条目，可选同时删除磁盘权重文件。"""
    registry = get_model_registry()
    cache = registry._cache.get(cache_id)  # type: ignore[attr-defined]
    if not cache:
        raise HTTPException(status_code=404, detail="缓存条目不存在")

    # 删除权重文件
    deleted_weight = False
    if delete_weight_file and cache.weight_path:
        try:
            weight_p = Path(cache.weight_path)
            if weight_p.exists() and weight_p.is_file():
                os.remove(weight_p)
                deleted_weight = True
        except OSError as e:
            logger.warning("Failed to delete weight file %s: %s", cache.weight_path, e)

    # 从注册表移除
    with registry._write_lock:  # type: ignore[attr-defined]
        registry._cache.pop(cache_id, None)  # type: ignore[attr-defined]
    registry._save_cache_index()  # type: ignore[attr-defined]

    return {
        "deleted": True,
        "cache_id": cache_id,
        "deleted_weight_file": deleted_weight,
    }
