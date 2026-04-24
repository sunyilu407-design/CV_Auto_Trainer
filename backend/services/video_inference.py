import base64
import cv2
from pathlib import Path
from ultralytics import YOLO


def annotate_video_frames(
    video_path: str,
    weights_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    max_frames: int = 30,
    class_names: list[str] | None = None,
) -> list[dict]:
    """
    用训练好的模型在视频帧上推理，返回带标注的帧和检测结果。

    返回格式：
    [
        {
            "frame_idx": int,
            "timestamp_ms": float,
            "frame_b64": str,           # 带标注的 JPEG base64
            "detections": [
                {"class": str, "conf": float, "bbox_xyxy": [x1,y1,x2,y2]}
            ]
        },
        ...
    ]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)

    model = YOLO(weights_path)
    labels = class_names or model.names

    results_list: list[dict] = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        detections: list[dict] = []

        preds = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        for r in preds:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = labels[cls_id] if isinstance(labels, dict) else labels[cls_id] if labels else f"class_{cls_id}"
                detections.append({
                    "class": label,
                    "conf": round(conf_val, 3),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                })
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buf).decode("ascii")
        results_list.append({
            "frame_idx": frame_idx,
            "timestamp_ms": round(timestamp_ms, 1),
            "frame_b64": frame_b64,
            "detections": detections,
        })
        saved += 1
        frame_idx += 1

    cap.release()
    return results_list
