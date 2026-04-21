"""
多模型训练编排器：按 training_priority 顺序训练 model_pipeline 中的所有需训练模型。
对已缓存的模型（reuse_cache_id）直接跳过训练，只使用缓存权重。

编排器在 TrainDispatcher 之上再封装一层：
1. 读取 task.algorithm_plan.model_pipeline
2. 过滤出 requires_training=true 的步骤
3. 按 training_priority 排序（小数字优先）
4. 对每一步：
    - 若 reuse_cache_id 存在 → 标记为复用，跳过训练
    - 否则用对应的 model_id / epochs / input_size 调用 TrainDispatcher
5. 把每步的结果汇总成 multi_model_artifacts
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from models.db import Task
from services.model_registry import get_model_registry, TrainedModelCache
from services.train_dispatcher import TrainDispatcher, TrainMode

logger = logging.getLogger(__name__)


TRAINABLE_ROLES = {"primary_detector", "secondary_detector", "classifier"}


class LocalMultiModelNotSupported(RuntimeError):
    """本地模式暂不支持多模型顺序训练（因为本地训练由前端 worker 驱动，无法循环调度）"""


class MultiModelTrainingOrchestrator:
    """按优先级顺序训练 model_pipeline 中的多个模型"""

    def __init__(self, db: Session):
        self.db = db
        self.dispatcher = TrainDispatcher(db)

    # ------------------------------------------------------------------
    # 规划阶段
    # ------------------------------------------------------------------

    def plan_training_steps(self, task: Task) -> List[Dict[str, Any]]:
        """
        根据 task.algorithm_plan.model_pipeline 生成训练步骤列表。
        每个步骤包含：step_id, model_id, role, training_priority,
                    requires_training, reuse_cache_id, epochs, input_size
        """
        plan = task.algorithm_plan or {}
        pipeline = plan.get("model_pipeline") or []

        steps: List[Dict[str, Any]] = []
        for step in pipeline:
            role = step.get("role")
            if role not in TRAINABLE_ROLES:
                continue
            if not step.get("requires_training", True) and not step.get("reuse_cache_id"):
                # 既不需要训练、也没有复用 → 跳过（纯粹是 pretrained 直接用）
                continue
            steps.append({
                "step_id": step.get("step_id") or f"{role}_{len(steps)}",
                "role": role,
                "model_id": step.get("recommended_model_id"),
                "training_priority": step.get("training_priority", 99),
                "requires_training": bool(step.get("requires_training", True)),
                "reuse_cache_id": step.get("reuse_cache_id"),
                "reuse_weight_path": step.get("reuse_weight_path"),
                "epochs": step.get("epochs"),
                "input_size": step.get("input_size"),
                "reason_zh": step.get("reason_zh", ""),
            })

        steps.sort(key=lambda s: s.get("training_priority", 99))
        return steps

    # ------------------------------------------------------------------
    # 执行阶段
    # ------------------------------------------------------------------

    def run(
        self,
        task: Task,
        base_train_config: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        按优先级顺序训练所有模型。返回形如：
        {
            "multi_model_artifacts": {
                "<step_id>": {
                    "source": "trained" | "reuse",
                    "model_id": "...",
                    "artifacts": { ... },  # 如果是 trained
                    "weight_path": "...",   # 如果是 reuse
                }
            },
            "primary_artifacts": { ... }   # 主检测器的产物，给 Delivery 阶段主打
        }
        """
        steps = self.plan_training_steps(task)
        if not steps:
            logger.info("No trainable steps in model_pipeline, skip orchestration")
            return {"multi_model_artifacts": {}, "primary_artifacts": {}}

        mode_str = base_train_config.get("train_mode", "cloud")
        if mode_str == "local" and len(steps) > 1:
            raise LocalMultiModelNotSupported(
                "本地训练模式暂不支持多模型顺序训练。请切换到云端训练，"
                "或只保留单一主检测器。"
            )

        mode = TrainMode.LOCAL if mode_str == "local" else TrainMode.CLOUD

        registry = get_model_registry()
        algorithm_plan = task.algorithm_plan or {}
        scenario_type = algorithm_plan.get("scenario_type", "")
        all_classes = [
            t.get("class_name", "")
            for t in algorithm_plan.get("targets", [])
            if t.get("class_name")
        ]

        multi_artifacts: Dict[str, Dict[str, Any]] = {}
        primary_artifacts: Dict[str, Any] = {}

        total = len(steps)
        for idx, step in enumerate(steps):
            step_id = step["step_id"]

            # ---- 1) plan 里已标记复用 ----
            if step.get("reuse_cache_id"):
                logger.info(
                    "Step %s plan-level reuse of cached model %s",
                    step_id, step["reuse_cache_id"],
                )
                self._record_reuse(multi_artifacts, step, step["reuse_cache_id"], step.get("reuse_weight_path"))
                if step["role"] == "primary_detector" and step.get("reuse_weight_path"):
                    primary_artifacts = {"best.pt": step["reuse_weight_path"]}
                self._notify(progress_callback, idx, total, step, "reuse")
                continue

            # ---- 2) 运行时检查：registry 里是否有可复用的已训练模型？
            # 这样即使上一次训练在第 N 步失败，重跑时前 N-1 步能直接复用缓存
            runtime_cached = registry.find_reusable_model(
                required_classes=all_classes,
                scenario_type=scenario_type,
            )
            if runtime_cached and self._is_cache_suitable(runtime_cached, step):
                logger.info(
                    "Step %s runtime-reuse of cached model %s",
                    step_id, runtime_cached.cache_id,
                )
                registry.increment_reuse(runtime_cached.cache_id)
                self._record_reuse(
                    multi_artifacts, step,
                    runtime_cached.cache_id, runtime_cached.weight_path,
                )
                if step["role"] == "primary_detector":
                    primary_artifacts = {"best.pt": runtime_cached.weight_path}
                self._notify(progress_callback, idx, total, step, "reuse")
                continue

            # ---- 3) 需要训练 ----
            per_model_config = dict(base_train_config)
            per_model_config["__orchestrated__"] = True  # 阻止 dispatcher 自己注册缓存
            if step.get("model_id"):
                per_model_config["model"] = step["model_id"]
            if step.get("epochs"):
                per_model_config["epochs"] = step["epochs"]
            if step.get("input_size"):
                per_model_config["imgsz"] = step["input_size"]

            def _wrapped_progress(s: Dict[str, Any], _idx=idx, _step=step):
                if progress_callback:
                    progress_callback({
                        **s,
                        "current_model_index": _idx,
                        "total_models": total,
                        "current_step_id": _step["step_id"],
                        "current_model_id": _step["model_id"],
                        "source": "trained",
                    })

            logger.info(
                "Training step %d/%d: %s (%s)",
                idx + 1, total, step_id, step["model_id"],
            )
            try:
                result = self.dispatcher.dispatch(
                    mode=mode,
                    task_id=task.id,
                    train_config=per_model_config,
                    progress_callback=_wrapped_progress,
                )
            except Exception:
                multi_artifacts[step_id] = {
                    "source": "failed",
                    "model_id": step["model_id"],
                    "role": step["role"],
                }
                task.training_progress = {
                    **(task.training_progress or {}),
                    "multi_model_artifacts": multi_artifacts,
                    "failed_step_id": step_id,
                }
                self.db.commit()
                raise

            multi_artifacts[step_id] = {
                "source": "trained",
                "model_id": step["model_id"],
                "role": step["role"],
                "artifacts": result,
            }
            if step["role"] == "primary_detector":
                primary_artifacts = result

            # 训练成功 → 注册缓存 (带 step_id，避免多步共享同一 model_id 时冲突)
            self._register_step_cache(task, step, per_model_config, result, all_classes, scenario_type)

        return {
            "multi_model_artifacts": multi_artifacts,
            "primary_artifacts": primary_artifacts,
        }

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _record_reuse(
        self,
        multi_artifacts: Dict[str, Dict[str, Any]],
        step: Dict[str, Any],
        cache_id: str,
        weight_path: Optional[str],
    ):
        multi_artifacts[step["step_id"]] = {
            "source": "reuse",
            "model_id": step["model_id"],
            "role": step["role"],
            "cache_id": cache_id,
            "weight_path": weight_path,
        }

    def _notify(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        idx: int,
        total: int,
        step: Dict[str, Any],
        source: str,
    ):
        if not progress_callback:
            return
        progress_callback({
            "current_model_index": idx,
            "total_models": total,
            "current_step_id": step["step_id"],
            "current_model_id": step["model_id"],
            "source": source,
            "done": False,
        })

    def _is_cache_suitable(self, cache: TrainedModelCache, step: Dict[str, Any]) -> bool:
        """判断缓存模型是否适合该步骤：角色/基础模型兼容"""
        # 必须基于相同或同族预训练模型
        if cache.source_model_id and step.get("model_id"):
            if cache.source_model_id != step["model_id"]:
                return False
        return True

    def _register_step_cache(
        self,
        task: Task,
        step: Dict[str, Any],
        train_config: Dict[str, Any],
        result: Dict[str, Any],
        all_classes: List[str],
        scenario_type: str,
    ):
        try:
            best_pt = result.get("best.pt") or result.get("best_weight")
            if not best_pt or not all_classes:
                return

            # cache_id 带 step_id 避免多步骤互相覆盖
            cache_id = f"{task.id}_{step['step_id']}_{train_config.get('model', 'unknown')}"
            cache = TrainedModelCache(
                cache_id=cache_id,
                source_model_id=train_config.get("model", "unknown"),
                task_id=task.id,
                classes=all_classes,
                class_count=len(all_classes),
                scenario_type=scenario_type,
                map50=getattr(task, "best_map50", None),
                map50_95=getattr(task, "best_map50_95", None),
                weight_path=str(best_pt),
                export_paths={k: v for k, v in result.items() if k != "best.pt" and isinstance(v, str)},
                trained_at=time.time(),
                image_count=getattr(task, "total_image_count", 0) or 0,
                epochs_completed=train_config.get("epochs", 0),
                tags=[scenario_type, step.get("role", "")],
            )
            get_model_registry().register_trained_model(cache)
            logger.info("Registered step cache: %s", cache_id)
        except Exception as e:
            logger.warning("Failed to register step cache for %s: %s", step.get("step_id"), e)
