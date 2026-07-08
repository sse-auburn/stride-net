"""
STRIDE-Net segmentation training script.

Trains the YOLOv8n-seg backbone used in STRIDE-Net for inter-row spacing
segmentation in ornamental nurseries. This script covers only the
segmentation model training. The downstream geometric refinement stages
are described in the associated publication.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train STRIDE-Net segmentation model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=0, help="CUDA device id or 'cpu'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--project", default="runs/segment", help="Output project dir")
    parser.add_argument("--name", default="stride_net", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        amp=True
    )


if __name__ == "__main__":
    main()
