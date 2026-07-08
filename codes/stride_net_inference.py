"""
STRIDE-Net: Segmentation Inference Module
------------------------------------------
This module provides the segmentation-stage inference for STRIDE-Net,
the hybrid perception framework for intra-bed navigation in ornamental
nurseries described in the associated publication.

This file documents the input/output specifications of the YOLOv8n-seg
segmentation backbone used in STRIDE-Net. The downstream geometric
refinement stages (contour extraction, PCA-based orientation filtering,
Hough line fitting, vanishing-point construction, and angular-deviation
estimation) are described conceptually in the manuscript and are not
included in this release.

Maintained by the Smart Systems Engineering (SSE) Laboratory,
Department of Biosystems Engineering, Auburn University.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class STRIDENetSegmenter:
    """
    Segmentation front-end of STRIDE-Net.

    Input:
        RGB image (H x W x 3, uint8). Internally resized/letterboxed to
        the training resolution of 640 x 640 before inference.

    Output:
        A list of binary instance masks (uint8, values 0 or 255), one per
        detected inter-row spacing region. These masks are the input to the
        geometric refinement stages described in the manuscript.
    """

    def __init__(self, weights_path, imgsz=640, conf=0.25, device=None):
        """
        Args:
            weights_path (str): Path to the trained YOLOv8n-seg checkpoint (.pt).
            imgsz (int): Inference input resolution (default 640, matching training).
            conf (float): Confidence threshold for mask predictions.
            device (str or int): 'cpu', 0 (GPU id), or None for auto-selection.
        """
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def segment(self, image):
        """
        Run segmentation on a single RGB image.

        Args:
            image (np.ndarray): RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            list[np.ndarray]: List of binary masks (H, W), dtype uint8,
                              one per detected spacing region.
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        masks_out = []
        if results and results[0].masks is not None:
            h, w = image.shape[:2]
            for m in results[0].masks:
                mask_np = (m.data[0].cpu().numpy() * 255).astype(np.uint8)
                # Resize mask back to original image size if needed
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h),
                                         interpolation=cv2.INTER_NEAREST)
                masks_out.append(mask_np)

        return masks_out

    def annotated(self, image):
        """
        Return the model's annotated visualization (masks overlaid on image).

        Args:
            image (np.ndarray): RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            np.ndarray: Annotated image with segmentation overlays.
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        return results[0].plot()
