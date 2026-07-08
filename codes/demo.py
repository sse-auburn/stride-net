"""
STRIDE-Net segmentation demo.

Runs the YOLOv8n-seg segmentation stage on a single input image and saves
the annotated output. The geometric refinement stages that convert these
masks into an angular-deviation control signal are described in the
associated publication.

Usage:
    python demo.py --weights weights/stride_seg_m.pt --image sample.jpg
"""

import argparse
import cv2
from stride_net_inference import STRIDENetSegmenter


def main():
    parser = argparse.ArgumentParser(description="STRIDE-Net segmentation demo")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8n-seg weights (.pt)")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default="stride_net_demo_output.jpg",
                        help="Path to save annotated output")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Initialize segmenter
    segmenter = STRIDENetSegmenter(
        weights_path=args.weights,
        imgsz=args.imgsz,
        conf=args.conf
    )

    # Run segmentation
    masks = segmenter.segment(image)
    print(f"Detected {len(masks)} spacing region(s).")

    # Save annotated visualization
    annotated = segmenter.annotated(image)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated output saved to: {args.output}")


if __name__ == "__main__":
    main()
