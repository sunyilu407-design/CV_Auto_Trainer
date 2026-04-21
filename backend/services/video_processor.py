"""
视频处理服务：上传视频拆帧、抽帧给 VLM、离线验证。

支持：
1. 视频上传后自动拆帧为图片（用于 VLM 分析和打标）
2. 智能抽帧：按关键帧 / 固定间隔 / 场景变化抽帧
3. 离线视频验证：对用户上传的验证视频使用 VLM 评估方案可行性
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    mode: str = "interval",
    interval_seconds: float = 1.0,
    max_frames: int = 200,
    target_size: tuple[int, int] | None = None,
) -> Dict[str, Any]:
    """
    从视频中提取帧。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        mode: 抽帧模式 - "interval" (固定间隔), "keyframe" (关键帧), "scene" (场景变化)
        interval_seconds: interval 模式的间隔秒数
        max_frames: 最大帧数
        target_size: 目标尺寸 (width, height)，None 保持原始尺寸

    Returns:
        { "frame_count": int, "frame_paths": list[str], "video_info": dict }
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("需要安装 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    video_info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_seconds, 2),
    }

    os.makedirs(output_dir, exist_ok=True)
    frame_paths: list[str] = []

    if mode == "interval":
        frame_interval = max(1, int(fps * interval_seconds))
    elif mode == "keyframe":
        frame_interval = max(1, int(fps * 2))
    else:
        frame_interval = max(1, int(fps * interval_seconds))

    frame_idx = 0
    saved_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if saved_count >= max_frames:
            break

        should_save = False
        if mode == "interval" or mode == "keyframe":
            should_save = (frame_idx % frame_interval == 0)
        elif mode == "scene":
            should_save = _is_scene_change(prev_frame, frame, threshold=30.0)
            if frame_idx == 0:
                should_save = True

        if should_save:
            if target_size:
                frame = cv2.resize(frame, target_size)

            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(filepath)
            saved_count += 1

        prev_frame = frame
        frame_idx += 1

    cap.release()

    return {
        "frame_count": saved_count,
        "frame_paths": frame_paths,
        "video_info": video_info,
    }


def _is_scene_change(prev_frame, curr_frame, threshold: float = 30.0) -> bool:
    """基于帧差检测场景变化"""
    if prev_frame is None:
        return True
    try:
        import cv2
        import numpy as np
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = float(np.mean(diff))
        return mean_diff > threshold
    except Exception:
        return False


def extract_frames_for_vlm(
    video_path: str,
    max_frames: int = 8,
    target_size: tuple[int, int] = (640, 480),
) -> List[str]:
    """
    从视频中抽取关键帧，直接返回 base64 编码列表，用于 VLM 分析。
    不写磁盘，全在内存完成。
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("需要安装 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("视频帧数为 0")

    indices = _compute_sample_indices(total_frames, max_frames)
    frames_base64: list[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, target_size)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_base64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    return frames_base64


def _compute_sample_indices(total: int, n: int) -> list[int]:
    """均匀采样 n 个帧索引"""
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(step * i) for i in range(n)]


def validate_video_with_vlm(
    video_path: str,
    algorithm_plan: Dict[str, Any],
    vlm_adapter,
    max_frames: int = 8,
    max_retry: int = 1,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    离线视频验证：用 VLM 评估用户上传的场景视频是否与算法方案匹配。

    Returns:
        {
            "validation_passed": bool,
            "confidence": float,
            "analysis_zh": str,      # 中文分析结果
            "suggestions_zh": list,   # 改进建议
            "frame_count_analyzed": int,
        }
    """
    frames_b64 = extract_frames_for_vlm(video_path, max_frames=max_frames)
    if not frames_b64:
        return {
            "validation_passed": False,
            "confidence": 0.0,
            "analysis_zh": "无法从视频中提取有效帧",
            "suggestions_zh": ["请检查视频文件是否损坏"],
            "frame_count_analyzed": 0,
        }

    targets = algorithm_plan.get("targets", [])
    events = algorithm_plan.get("events", [])
    scenario = algorithm_plan.get("scenario_type", "custom")
    summary = algorithm_plan.get("summary_zh") or algorithm_plan.get("summary", "")

    target_names = [t.get("display_name_zh") or t.get("class_name", "") for t in targets]
    event_names = [e.get("name_zh") or e.get("name", "") for e in events]

    system_prompt = """你是一位计算机视觉方案验证专家。你的任务是分析用户提供的实际场景视频帧，
判断这些画面是否与预定的算法方案匹配。

请用中文回答，输出 JSON 格式：
{
  "validation_passed": true/false,
  "confidence": 0.0-1.0,
  "analysis_zh": "一段简要分析",
  "suggestions_zh": ["建议1", "建议2"],
  "detected_objects": ["看到的对象列表"],
  "scene_description_zh": "场景描述"
}"""

    user_text = f"""## 算法方案摘要
{summary}

## 需要检测的目标
{', '.join(target_names)}

## 预期事件
{', '.join(event_names)}

## 场景类型
{scenario}

请分析这些视频帧中是否包含上述目标和场景，并评估算法方案的可行性。"""

    try:
        import json
        raw = vlm_adapter.call_with_system_prompt(
            system_prompt=system_prompt,
            user_text=user_text,
            images_base64=frames_b64,
            response_format="json",
            max_tokens=max_tokens,
            max_retry=max_retry,
        )
        result = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        result["frame_count_analyzed"] = len(frames_b64)
        return result
    except Exception as e:
        logger.exception("Video validation VLM call failed")
        return {
            "validation_passed": None,
            "confidence": 0.0,
            "analysis_zh": f"VLM 验证失败: {str(e)[:200]}",
            "suggestions_zh": ["请检查 VLM 服务是否正常配置"],
            "frame_count_analyzed": len(frames_b64),
        }


def get_video_info(video_path: str) -> Dict[str, Any]:
    """获取视频基本信息"""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("需要安装 opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": round(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)), 2
        ),
    }
    cap.release()
    return info
