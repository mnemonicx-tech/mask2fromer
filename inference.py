"""
inference.py
------------
Load a trained Mask2Former model and run instance segmentation on images.

Features
  • Single-image inference with PIL / OpenCV input
  • Handles multiple overlapping clothing layers (sorted by depth/area)
  • Colour-coded per-class masks with alpha blending
  • Side-by-side visualisation saved to disk
  • Configurable score threshold and max detections
  • JSON export of predictions

Usage:
    # Single image
    python inference.py --image /path/to/photo.jpg \
                        --weights ./output/model_final.pth \
                        --output-dir ./results

    # Batch of images in a folder
    python inference.py --image-dir /path/to/photos/ \
                        --weights ./output/model_final.pth \
                        --output-dir ./results \
                        --score-threshold 0.5
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import detectron2.data.transforms as T
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer

from register_dataset import register_fashion_datasets, FASHION_CLASSES
from config_setup import build_cfg

logger = logging.getLogger("mask2former.inference")


# ---------------------------------------------------------------------------
# Colour palette — one distinct colour per class (HSV → RGB)
# ---------------------------------------------------------------------------

def _build_colour_palette(n: int) -> np.ndarray:
    """Return (n, 3) uint8 array of visually distinct BGR colours."""
    palette = []
    for i in range(n):
        hue = int(180 * i / n)
        colour = cv2.cvtColor(
            np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        palette.append(colour.tolist())
    return np.array(palette, dtype=np.uint8)


COLOUR_PALETTE = _build_colour_palette(len(FASHION_CLASSES))


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class FashionPredictor:
    """
    Wraps a trained Mask2Former model for single-pass inference.

    Parameters
    ----------
    cfg            : Detectron2 CfgNode (can be frozen).
    weights_path   : Path to ``model_final.pth`` or any checkpoint.
    score_threshold: Discard predictions below this confidence.
    """

    def __init__(
        self,
        cfg: CfgNode,
        weights_path: str,
        score_threshold: float = 0.5,
    ):
        self.cfg = cfg.clone()
        self.score_threshold = score_threshold

        # Build & load model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weights_path)
        logger.info(f"Loaded weights from: {weights_path}")

        # Pre-processing transform
        self.aug = T.ResizeShortestEdge(
            short_edge_length=[cfg.INPUT.MIN_SIZE_TEST],
            max_size=cfg.INPUT.MAX_SIZE_TEST,
            sample_style="choice",
        )
        self.input_format = cfg.INPUT.FORMAT  # "RGB" or "BGR"

        # Class metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> Instances:
        """
        Run inference on a single BGR numpy image (H×W×3 uint8).
        Returns Detectron2 Instances on CPU.
        """
        h, w = image_bgr.shape[:2]

        if self.input_format == "RGB":
            image = image_bgr[:, :, ::-1]
        else:
            image = image_bgr

        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        transform = self.aug.get_transform(image)
        image_tensor = torch.as_tensor(
            transform.apply_image(image).astype("float32").transpose(2, 0, 1)
        )

        inputs = [{"image": image_tensor, "height": h, "width": w}]
        outputs = self.model(inputs)[0]

        instances: Instances = outputs["instances"].to("cpu")

        # Filter by score threshold
        keep = instances.scores >= self.score_threshold
        return instances[keep]

    def predict_file(self, image_path: str) -> Tuple[np.ndarray, Instances]:
        """Load image from disk and run prediction. Returns (bgr_image, instances)."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return bgr, self.predict(bgr)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_predictions(
    bgr_image: np.ndarray,
    instances: Instances,
    metadata,
    alpha: float = 0.5,
    sort_by_area: bool = True,
    max_detections: int = 50,
) -> np.ndarray:
    """
    Render coloured masks blended onto the original image.

    Masks are drawn largest → smallest so smaller garments appear on top
    (handles layered clothing, e.g. shirt under jacket).

    Returns BGR image with overlaid masks.
    """
    output = bgr_image.copy().astype(np.float32)
    n = min(len(instances), max_detections)

    if n == 0:
        return bgr_image

    masks   = instances.pred_masks[:n].numpy()    # (N, H, W) bool
    classes = instances.pred_classes[:n].numpy()  # (N,) int
    scores  = instances.scores[:n].numpy()        # (N,) float

    if sort_by_area:
        areas = masks.sum(axis=(1, 2))
        order = np.argsort(-areas)                # descending area → paint big first
        masks, classes, scores = masks[order], classes[order], scores[order]

    overlay = output.copy()
    for mask, cls_id, score in zip(masks, classes, scores):
        colour = COLOUR_PALETTE[cls_id % len(COLOUR_PALETTE)].astype(np.float32)
        overlay[mask] = colour

    # Alpha blend
    blended = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

    # Draw per-instance labels above mask centroid
    for mask, cls_id, score in zip(masks, classes, scores):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = f"{FASHION_CLASSES[cls_id]}: {score:.2f}"
        # Shadow
        cv2.putText(blended, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        # Text
        cv2.putText(blended, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return blended.astype(np.uint8)


def save_side_by_side(
    original: np.ndarray,
    rendered: np.ndarray,
    output_path: str,
) -> None:
    """Save original and prediction side-by-side."""
    combined = np.hstack([original, rendered])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, combined)
    logger.info(f"Saved visualisation: {output_path}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def instances_to_json(
    instances: Instances,
    image_path: str,
) -> Dict:
    """Convert Instances to a JSON-serialisable dict."""
    records = []
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy().astype(bool)
        # RLE encode mask to keep JSON compact
        from pycocotools import mask as mask_util
        rle = mask_util.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")

        records.append({
            "class_id":    int(instances.pred_classes[i]),
            "class_name":  FASHION_CLASSES[int(instances.pred_classes[i])],
            "score":       float(instances.scores[i]),
            "bbox":        instances.pred_boxes[i].tensor[0].tolist()
                           if instances.has("pred_boxes") else None,
            "mask_rle":    rle,
        })

    return {
        "image":       image_path,
        "num_detected": len(records),
        "predictions": records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_single(
    predictor: FashionPredictor,
    image_path: str,
    output_dir: str,
    save_json: bool = True,
    alpha: float = 0.5,
    max_detections: int = 50,
) -> Dict:
    """Predict, visualise, and optionally save JSON for one image."""
    t0 = time.perf_counter()
    bgr, instances = predictor.predict_file(image_path)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"{os.path.basename(image_path)}: "
        f"{len(instances)} detections in {elapsed * 1000:.1f} ms"
    )

    rendered = render_predictions(
        bgr,
        instances,
        predictor.metadata,
        alpha=alpha,
        max_detections=max_detections,
    )

    stem = Path(image_path).stem
    vis_path = os.path.join(output_dir, f"{stem}_pred.jpg")
    save_side_by_side(bgr, rendered, vis_path)

    result = instances_to_json(instances, image_path)

    if save_json:
        json_path = os.path.join(output_dir, f"{stem}_pred.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask2Former fashion inference")
    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     help="Path to a single image file")
    group.add_argument("--image-dir", help="Directory of images to process in batch")

    # Model
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained model weights (model_final.pth)",
    )
    parser.add_argument(
        "--backbone",
        default="R50",
        choices=["R50", "SWIN_T"],
    )
    parser.add_argument(
        "--output-dir",
        default="./inference_results",
        help="Where to save visualisations and JSON",
    )

    # Inference options
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to keep a prediction",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay opacity (0=transparent, 1=opaque)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Maximum number of instances to render",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON prediction files",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    register_fashion_datasets()

    cfg = build_cfg(
        output_dir=args.output_dir,
        backbone=args.backbone,
    )

    predictor = FashionPredictor(
        cfg,
        weights_path=args.weights,
        score_threshold=args.score_threshold,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if args.image:
        images = [args.image]
    else:
        images = [
            str(p)
            for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in supported_ext
        ]
        images.sort()
        logger.info(f"Found {len(images)} images in {args.image_dir}")

    for img_path in images:
        try:
            process_single(
                predictor,
                img_path,
                output_dir=args.output_dir,
                save_json=not args.no_json,
                alpha=args.alpha,
                max_detections=args.max_detections,
            )
        except Exception as exc:
            logger.error(f"Failed on {img_path}: {exc}", exc_info=True)

    logger.info(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
