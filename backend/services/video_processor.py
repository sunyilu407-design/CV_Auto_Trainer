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
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    从视频中抽取关键帧，直接返回 base64 编码列表，用于 VLM 分析。
    不写磁盘，全在内存完成。

    Returns:
        (frames_base64: list[str], frame_meta: list[dict])
        frame_meta 每项包含：{ "frame_index": int, "timestamp_ms": int, "source": "keyframe"|"uniform" }
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
    if total_frames <= 0:
        cap.release()
        raise ValueError("视频帧数为 0")

    indices, sources = _compute_hybrid_sample_indices(video_path, total_frames, max_frames, fps)
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
    return frames_base64, sources


def _compute_hybrid_sample_indices(
    video_path: str,
    total: int,
    n: int,
    fps: float,
) -> tuple[List[int], List[Dict[str, Any]]]:
    """
    混合采样：50% 关键帧 + 50% 均匀采样。
    先用 ffmpeg 提取关键帧列表（如果可用），否则回退到帧差分启发式。
    """
    if total <= n:
        meta = [{"frame_index": i, "timestamp_ms": int(i / max(fps, 1) * 1000), "source": "uniform"} for i in range(total)]
        return list(range(total)), meta

    keyframe_count = n // 2
    uniform_count = n - keyframe_count

    # 尝试 ffmpeg 关键帧提取
    keyframe_indices = _ffmpeg_extract_keyframes(video_path, total, keyframe_count)
    if keyframe_indices is None:
        # 回退：基于帧差分的启发式关键帧提取
        keyframe_indices = _frame_diff_keyframes(video_path, total, keyframe_count)

    if not keyframe_indices:
        # 最终回退：纯均匀采样
        step = total / n
        indices = [int(step * i) for i in range(n)]
        meta = [{"frame_index": idx, "timestamp_ms": int(idx / max(fps, 1) * 1000), "source": "uniform"} for idx in indices]
        return indices, meta

    # 合并：取 keyframe_count 个关键帧
    keyframe_set = set(keyframe_indices[:keyframe_count])

    # 从非关键帧区间做均匀采样
    non_keyframe = [i for i in range(total) if i not in keyframe_set]
    if not non_keyframe:
        step = total / n
        indices = [int(step * i) for i in range(n)]
        meta = [{"frame_index": idx, "timestamp_ms": int(idx / max(fps, 1) * 1000), "source": "uniform"} for idx in indices]
        return indices, meta

    step = len(non_keyframe) / uniform_count
    uniform_indices = [non_keyframe[min(int(step * i), len(non_keyframe) - 1)] for i in range(uniform_count)]

    # 合并并按时间排序
    combined = sorted(set(keyframe_indices[:keyframe_count] + uniform_indices))
    # 如果合并后不足 n，补充均匀采样
    if len(combined) < n:
        remaining = [i for i in range(total) if i not in set(combined)]
        step = len(remaining) / (n - len(combined))
        for i in range(n - len(combined)):
            idx = remaining[min(int(step * i), len(remaining) - 1)]
            if idx not in combined:
                combined.append(idx)
        combined.sort()

    meta = []
    for idx in combined:
        src = "keyframe" if idx in keyframe_set else "uniform"
        meta.append({"frame_index": idx, "timestamp_ms": int(idx / max(fps, 1) * 1000), "source": src})

    return combined, meta


def _ffmpeg_extract_keyframes(video_path: str, total: int, n: int) -> List[int] | None:
    """调用 ffmpeg 提取关键帧索引，失败返回 None"""
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path,
             "-vf", "select='eq(pict_type,PICT_TYPE_I)'",
             "-vsync", "vfr", "-f", "mkvpipe", "-"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            return None

        # 解析 ffmpeg 帧索引输出（ffmpeg -frame_pts 输出帧编号）
        pts_output = subprocess.run(
            ["ffmpeg", "-i", video_path,
             "-vf", "select='eq(pict_type,PICT_TYPE_I)',showinfo",
             "-vsync", "vfr", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        indices = []
        for line in pts_output.stderr.splitlines():
            if "pts_time" in line:
                try:
                    ts = float(line.split("pts_time:")[1].split()[0])
                    frame_idx = int(ts * 30)  # 估算帧号
                    if 0 <= frame_idx < total:
                        indices.append(frame_idx)
                except (ValueError, IndexError):
                    continue

        if indices:
            return sorted(set(indices[:n * 3]))  # 多取一些，后面截断
        return None

    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _frame_diff_keyframes(video_path: str, total: int, n: int) -> List[int]:
    """基于帧差分的启发式关键帧提取（无需 ffmpeg）"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    prev_gray = None
    diff_scores: List[tuple[int, float]] = []

    frame_idx = 0
    sample_step = max(1, total // (n * 5))  # 采样密度
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            diff_scores.append((frame_idx, diff))
        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not diff_scores:
        return []

    # 取帧差最大的 n 个作为关键帧
    diff_scores.sort(key=lambda x: x[1], reverse=True)
    keyframe_indices = sorted([idx for idx, _ in diff_scores[:n * 2]])

    # 确保均匀分布（避免关键帧扎堆）
    if len(keyframe_indices) > n:
        step = len(keyframe_indices) / n
        keyframe_indices = [keyframe_indices[min(int(step * i), len(keyframe_indices) - 1)] for i in range(n)]

    return keyframe_indices


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
    frames_b64, frame_meta = extract_frames_for_vlm(video_path, max_frames=max_frames)
    if not frames_b64:
        return {
            "validation_passed": False,
            "confidence": 0.0,
            "analysis_zh": "无法从视频中提取有效帧",
            "suggestions_zh": ["请检查视频文件是否损坏"],
            "frame_count_analyzed": 0,
        }

    # 提取视频元数据用于 VLM 上下文
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = round(video_total / max(video_fps, 1), 2)
    cap.release()
    video_info = {
        "fps": video_fps, "total_frames": video_total,
        "width": video_width, "height": video_height,
        "duration_seconds": video_duration,
    }

    targets = algorithm_plan.get("targets", [])
    events = algorithm_plan.get("events", [])
    scenario = algorithm_plan.get("scenario_type", "custom")
    summary = algorithm_plan.get("summary_zh") or algorithm_plan.get("summary", "")

    target_names = [t.get("display_name_zh") or t.get("class_name", "") for t in targets]
    event_names = [e.get("name_zh") or e.get("name", "") for e in events]

    system_prompt = f"""你是一位计算机视觉方案验证专家。你的任务是分析用户提供的实际场景视频帧，
判断这些画面是否与预定的算法方案匹配。

## 视频上下文信息
- 视频帧率: {video_info['fps']:.1f} fps
- 视频时长: {video_info['duration_seconds']:.1f} 秒
- 视频分辨率: {video_info['width']}x{video_info['height']}
- 采样帧数: {len(frames_b64)} 帧（关键帧 + 均匀采样混合）
请结合视频的时间跨度理解画面变化，避免将某一帧的瞬时状态误判为常态。

请用中文回答，输出 JSON 格式：
{{
  "validation_passed": true/false,
  "confidence": 0.0-1.0,
  "analysis_zh": "一段简要分析",
  "suggestions_zh": ["建议1", "建议2"],
  "detected_objects": ["看到的对象列表"],
  "scene_description_zh": "场景描述"
}}"""

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
