"""
Evaluate the trained STRIDE-Net segmentation model.

Reports standard Ultralytics segmentation metrics (precision, recall, mAP)
on the validation/test split defined in data.yaml.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate STRIDE-Net segmentation model")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(metrics)


if __name__ == "__main__":
    main()
