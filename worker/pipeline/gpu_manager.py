import torch
import gc
import threading
from contextlib import contextmanager

_current_stage = {"name": None, "cancelled": False, "lock": threading.Lock()}


def get_device() -> str:
    """获取可用的计算设备，优先级：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_free_memory_gb() -> float:
    """获取当前设备的可用内存（GB）"""
    device = get_device()
    if device == "cuda":
        return (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1e9
    elif device == "mps":
        # MPS 不提供精确的内存查询，使用系统工具估算
        import psutil
        return psutil.virtual_memory().available / 1e9
    else:
        import psutil
        return psutil.virtual_memory().available / 1e9


class CancelError(Exception):
    """推理循环中检测到取消标志时抛出"""
    pass


@contextmanager
def gpu_stage(stage_name: str, required_gb: float = 2.0):
    """
    显存安全上下文管理器。
    进入时检查显存是否充足，退出时强制释放。
    支持取消：外部设置 _current_stage['cancelled'] = True 时，
    推理循环检测到标志后主动跳出并释放资源。
    """
    device = get_device()

    if device in ("cuda", "mps"):
        free_gb = get_free_memory_gb()
        if free_gb < required_gb:
            raise MemoryError(
                f"阶段 [{stage_name}] 需要 {required_gb:.1f}GB 显存，"
                f"当前仅剩 {free_gb:.1f}GB（设备: {device}）"
            )

    with _current_stage["lock"]:
        _current_stage["name"] = stage_name
        _current_stage["cancelled"] = False

    try:
        yield
    finally:
        with _current_stage["lock"]:
            _current_stage["name"] = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.empty_cache()


def is_cancelled() -> bool:
    with _current_stage["lock"]:
        return _current_stage["cancelled"]


def cancel_current_stage():
    """收到 cancel 命令时调用，设置取消标志"""
    with _current_stage["lock"]:
        _current_stage["cancelled"] = True


def check_cancel_and_yield():
    """推理循环中周期性调用：检测到取消标志则抛出 CancelError"""
    if is_cancelled():
        raise CancelError("Stage cancelled by user")
