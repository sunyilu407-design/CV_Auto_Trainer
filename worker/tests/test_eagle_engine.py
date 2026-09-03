"""
Eagle 引擎测试脚本

测试 LocateAnything 和 Eagle2.5 适配器的基本功能
"""

import sys
import os

# 添加 worker 目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from pipeline.locate_anything_adapter import LocateAnythingAdapter
        print("[OK] LocateAnythingAdapter 导入成功")
    except ImportError as e:
        print(f"[WARN] LocateAnythingAdapter 导入失败: {e}")
        print("       这是正常的，如果没有安装 Eagle 依赖")
    
    try:
        from pipeline.eagle_vqa_adapter import EagleVqaAdapter
        print("[OK] EagleVqaAdapter 导入成功")
    except ImportError as e:
        print(f"[WARN] EagleVqaAdapter 导入失败: {e}")
        print("       这是正常的，如果没有安装 Eagle 依赖")
    
    try:
        from pipeline.output_converter import (
            xywhn_to_xyxy,
            xyxy_to_xywhn,
            compute_iou,
            deduplicate_boxes,
        )
        print("[OK] output_converter 导入成功")
    except ImportError as e:
        print(f"[FAIL] output_converter 导入失败: {e}")
        return False
    
    try:
        from pipeline.engine_router import (
            select_detection_engine,
            select_vqa_engine,
            get_engine_info,
            can_run_locate_anything,
            can_run_eagle_vqa,
        )
        print("[OK] engine_router 导入成功")
    except ImportError as e:
        print(f"[FAIL] engine_router 导入失败: {e}")
        return False
    
    return True


def test_output_converter():
    """测试输出格式转换工具"""
    print("\n" + "=" * 60)
    print("测试 2: 输出格式转换")
    print("=" * 60)
    
    try:
        from pipeline.output_converter import (
            xywhn_to_xyxy,
            xyxy_to_xywhn,
            compute_iou,
            deduplicate_boxes,
            standardize_detection,
        )
        
        # 测试 xywhn_to_xyxy
        bbox_xywhn = [0.5, 0.5, 0.2, 0.3]
        xyxy = xywhn_to_xyxy(bbox_xywhn)
        print(f"  xywhn_to_xyxy: {bbox_xywhn} -> {xyxy}")
        assert abs(xyxy["x1"] - 0.4) < 0.001, "x1 计算错误"
        assert abs(xyxy["y1"] - 0.35) < 0.001, "y1 计算错误"
        print("  [OK] xywhn_to_xyxy 计算正确")
        
        # 测试 xyxy_to_xywhn
        bbox2 = xyxy_to_xywhn(xyxy["x1"], xyxy["y1"], xyxy["x2"], xyxy["y2"])
        print(f"  xyxy_to_xywhn: {bbox2}")
        assert abs(bbox2[0] - bbox_xywhn[0]) < 0.001, "cx 计算错误"
        assert abs(bbox2[1] - bbox_xywhn[1]) < 0.001, "cy 计算错误"
        print("  [OK] xyxy_to_xywhn 计算正确")
        
        # 测试 compute_iou
        box1 = [0.5, 0.5, 0.2, 0.2]
        box2 = [0.5, 0.5, 0.1, 0.1]
        iou = compute_iou(box1, box2)
        print(f"  compute_iou: {box1} vs {box2} = {iou:.4f}")
        assert 0.25 <= iou <= 1.0, f"IoU 计算错误: {iou}"
        print("  [OK] compute_iou 计算正确")
        
        # 测试 deduplicate_boxes
        boxes = [
            {"class_name": "person", "bbox_xywhn": [0.5, 0.5, 0.2, 0.2], "conf": 0.9},
            {"class_name": "person", "bbox_xywhn": [0.5, 0.5, 0.19, 0.19], "conf": 0.8},  # 重复
            {"class_name": "car", "bbox_xywhn": [0.3, 0.3, 0.1, 0.1], "conf": 0.7},
        ]
        deduped = deduplicate_boxes(boxes, iou_threshold=0.5)
        print(f"  deduplicate_boxes: {len(boxes)} -> {len(deduped)}")
        assert len(deduped) == 2, f"去重后应有 2 个框，实际 {len(deduped)}"
        print("  [OK] deduplicate_boxes 工作正常")
        
        print("\n[SUCCESS] 输出格式转换测试通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 输出格式转换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_router():
    """测试引擎路由"""
    print("\n" + "=" * 60)
    print("测试 3: 引擎路由")
    print("=" * 60)
    
    try:
        from pipeline.engine_router import (
            select_detection_engine,
            select_vqa_engine,
            get_engine_info,
            can_run_locate_anything,
            can_run_eagle_vqa,
        )
        
        # 测试引擎选择
        classes = [
            {"class_name": "person"},
            {"class_name": "car"},
        ]
        
        result = select_detection_engine(classes, user_preference="auto")
        print(f"  自动检测引擎选择: {result['engine']}")
        print(f"    原因: {result['reason']}")
        print(f"    可用: {result['available']}")
        print(f"    需要显存: {result.get('vram_required_gb', 'N/A')} GB")
        
        # 测试 VQA 引擎选择
        result = select_vqa_engine(user_preference="auto")
        print(f"\n  自动 VQA 引擎选择: {result['engine']}")
        print(f"    原因: {result['reason']}")
        print(f"    可用: {result['available']}")
        
        # 测试引擎信息
        info = get_engine_info()
        print(f"\n  系统信息:")
        print(f"    CUDA 可用: {info['system']['has_cuda']}")
        if info['system']['has_cuda']:
            print(f"    GPU: {info['system'].get('gpu_name', 'N/A')}")
            print(f"    显存: {info['system'].get('free_vram_gb', 'N/A'):.1f} GB")
        
        print(f"\n  检测引擎:")
        for key, eng in info['detection_engines'].items():
            print(f"    {eng['name']}: 可用={eng['available']}")
        
        print(f"\n  VQA 引擎:")
        for key, eng in info['vqa_engines'].items():
            print(f"    {eng['name']}: 可用={eng['available']}")
        
        print("\n[SUCCESS] 引擎路由测试通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 引擎路由测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置模块"""
    print("\n" + "=" * 60)
    print("测试 4: 配置模块")
    print("=" * 60)
    
    try:
        from config import (
            WorkerConfig,
            DetectionEngine,
            VqaEngine,
            get_config,
            print_engine_status,
        )
        
        # 测试配置加载
        config = get_config()
        print(f"  引擎偏好: {config.engine_preference}")
        print(f"  LocateAnything 配置:")
        print(f"    模型: {config.locate_anything.model_id}")
        print(f"    置信度阈值: {config.locate_anything.conf_threshold}")
        print(f"  Eagle VQA 配置:")
        print(f"    模型: {config.eagle_vqa.model_id}")
        print(f"    质量阈值: {config.eagle_vqa.quality_threshold}")
        
        print("\n[SUCCESS] 配置模块测试通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 配置模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """打印测试摘要"""
    print("\n" + "=" * 60)
    print("Eagle 引擎集成测试摘要")
    print("=" * 60)
    print("""
新增文件:
  - worker/pipeline/locate_anything_adapter.py  (LocateAnything 封装器)
  - worker/pipeline/eagle_vqa_adapter.py      (Eagle2.5 VQA 封装器)
  - worker/pipeline/output_converter.py         (输出格式转换工具)
  - worker/config.py                           (配置管理)

修改文件:
  - worker/pipeline/engine_router.py            (添加 Eagle 路由支持)
  - worker/pipeline/stage2_labeler.py           (集成新模型)
  - backend/models/db.py                       (添加配置字段)
  - frontend/src/store/settingsStore.ts         (添加前端配置)

使用方式:
  # 自动选择 (根据显存可用性)
  run_detection_with_engine(image_dir, classes, engine="auto")
  
  # 强制使用 LocateAnything
  run_detection_with_engine(image_dir, classes, engine="locate_anything")
  
  # 强制使用 Eagle2.5 VQA
  run_quality_check_with_engine(raw_boxes_path, engine="eagle_vqa")

硬件要求:
  - LocateAnything: >= 6GB GPU 显存 (FP16)
  - Eagle2.5 VQA:  >= 16GB GPU 显存 (FP16)

注意: 当前电脑没有 NVIDIA GPU，测试将在没有 GPU 时优雅降级
    """)


def main():
    print("Eagle 引擎集成测试")
    print("=" * 60)
    print()
    
    # 运行测试
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_output_converter():
        all_passed = False
    
    if not test_engine_router():
        all_passed = False
    
    if not test_config():
        all_passed = False
    
    print_summary()
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed (may be due to missing GPU or dependencies)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
