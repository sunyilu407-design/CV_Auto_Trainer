#!/usr/bin/env python3
"""
模型导出脚本。
支持 ONNX / TensorRT / CoreML / OpenVINO。
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="best.pt path")
    parser.add_argument("--format", type=str, required=True, help="onnx/engine/coreml/openvino")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = model.export(format=args.format)
    print(f"Exported to: {exported}")


if __name__ == "__main__":
    main()
