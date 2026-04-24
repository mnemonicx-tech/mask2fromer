"""
edge_diagnostic.py — Visual edge failure analysis.
Generates zoomed-in boundary comparison images showing exactly where
predictions bleed, miss thin structures, or over-smooth edges.

Saves to ./edge_diagnostic/ for manual inspection before implementing boundary loss.

Usage:
    python edge_diagnostic.py --weights ./output_swin_fashion/model_0039999.pth --num-images 20
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog

from config_setup import build_cfg
from register_dataset import register_fashion_datasets, get_thing_classes
from validation_utils import ValidationDatasetMapper, get_boundaries, compute_bfscore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def make_edge_comparison(img_tensor, pred_masks, gt_masks, score_threshold=0.5):
    """Generate a multi-panel edge diagnostic image.
    
    Panels:
    1. Original image
    2. GT boundary overlay (green)
    3. Pred boundary overlay (red)  
    4. Error map (blue=missing, red=extra, green=correct)
    5. Zoomed edge region
    """
    # Normalize image
    img = img_tensor.cpu().float()
    if img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    C, H, W = img.shape
    
    # Compute boundaries
    k = 3  # strict boundary kernel
    
    gt_boundary = torch.zeros(H, W, device=gt_masks.device)
    pred_boundary = torch.zeros(H, W, device=pred_masks.device)
    
    if len(gt_masks) > 0:
        gt_b = get_boundaries(gt_masks.float(), k)
        gt_boundary = (torch.clamp(gt_b.sum(0), 0, 1) > 0).float()
    
    if len(pred_masks) > 0:
        pred_b = get_boundaries(pred_masks.float(), k)
        pred_boundary = (torch.clamp(pred_b.sum(0), 0, 1) > 0).float()
    
    # Resize boundaries to match image
    gt_b_resized = F.interpolate(
        gt_boundary.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest'
    ).squeeze().cpu()
    pred_b_resized = F.interpolate(
        pred_boundary.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest'
    ).squeeze().cpu()
    
    # Compute foreground masks
    gt_fg = (gt_masks.sum(0) > 0).float()
    pred_fg = (pred_masks.sum(0) > 0).float()
    
    gt_fg_resized = F.interpolate(
        gt_fg.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest'
    ).squeeze().cpu()
    pred_fg_resized = F.interpolate(
        pred_fg.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest'
    ).squeeze().cpu()
    
    # Panel 1: Original
    panel_orig = img.clone()
    
    # Panel 2: GT boundaries (green on image)
    panel_gt = img.clone()
    panel_gt[1][gt_b_resized > 0] = 1.0
    panel_gt[0][gt_b_resized > 0] = 0.0
    panel_gt[2][gt_b_resized > 0] = 0.0
    
    # Panel 3: Pred boundaries (red on image)
    panel_pred = img.clone()
    panel_pred[0][pred_b_resized > 0] = 1.0
    panel_pred[1][pred_b_resized > 0] = 0.0
    panel_pred[2][pred_b_resized > 0] = 0.0
    
    # Panel 4: Error analysis
    # Green = both correct (boundary matches)
    # Red = prediction bleeding (pred boundary but no GT)
    # Blue = missing (GT boundary but no prediction)
    panel_error = img.clone() * 0.3  # dim background
    
    correct = (gt_b_resized > 0) & (pred_b_resized > 0)
    missing = (gt_b_resized > 0) & (pred_b_resized == 0)
    extra = (pred_b_resized > 0) & (gt_b_resized == 0)
    
    panel_error[1][correct] = 1.0   # green = matched
    panel_error[2][missing] = 1.0   # blue = missing prediction
    panel_error[0][extra] = 1.0     # red = prediction bleeding
    
    # Panel 5: Mask overlap analysis
    # Green = correct coverage, Red = over-prediction, Blue = under-prediction
    panel_mask = img.clone() * 0.3
    mask_correct = (gt_fg_resized > 0) & (pred_fg_resized > 0)
    mask_under = (gt_fg_resized > 0) & (pred_fg_resized == 0)
    mask_over = (pred_fg_resized > 0) & (gt_fg_resized == 0)
    
    panel_mask[1][mask_correct] = 0.7
    panel_mask[2][mask_under] = 1.0
    panel_mask[0][mask_over] = 1.0
    
    # Compute per-image metrics
    bfscore = compute_bfscore(pred_b_resized > 0, gt_b_resized > 0, threshold_px=3)
    
    n_missing = missing.sum().item()
    n_extra = extra.sum().item()
    n_correct = correct.sum().item()
    n_gt_total = (gt_b_resized > 0).sum().item()
    
    metrics = {
        'bfscore': bfscore,
        'missing_px': n_missing,
        'extra_px': n_extra,
        'correct_px': n_correct,
        'gt_boundary_px': n_gt_total,
        'recall': n_correct / max(n_gt_total, 1),
        'precision': n_correct / max(n_correct + n_extra, 1),
        'mask_under_px': mask_under.sum().item(),
        'mask_over_px': mask_over.sum().item(),
    }
    
    return [panel_orig, panel_gt, panel_pred, panel_error, panel_mask], metrics


def tensor_to_pil(tensor_chw):
    """Convert [C, H, W] float tensor to PIL Image."""
    arr = (tensor_chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_diagnostic(panels, metrics, idx, output_dir, category="unknown"):
    """Save a multi-panel diagnostic image with metrics annotation."""
    pil_panels = [tensor_to_pil(p) for p in panels]
    
    labels = ["Original", "GT Boundary", "Pred Boundary", "Edge Error", "Mask Error"]
    
    # Scale panels to same height
    target_h = 400
    resized = []
    for p in pil_panels:
        ratio = target_h / p.height
        new_w = int(p.width * ratio)
        resized.append(p.resize((new_w, target_h), Image.LANCZOS))
    
    # Create composite
    total_w = sum(p.width for p in resized) + 10 * (len(resized) - 1)
    canvas = Image.new('RGB', (total_w, target_h + 80), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    
    x_offset = 0
    for i, (panel, label) in enumerate(zip(resized, labels)):
        canvas.paste(panel, (x_offset, 0))
        # Label
        draw.text((x_offset + 5, target_h + 5), label, fill=(200, 200, 200))
        x_offset += panel.width + 10
    
    # Metrics text
    metrics_text = (
        f"BFScore: {metrics['bfscore']:.3f}  |  "
        f"Boundary Recall: {metrics['recall']:.3f}  |  "
        f"Boundary Precision: {metrics['precision']:.3f}  |  "
        f"Missing: {metrics['missing_px']}px  |  "
        f"Bleeding: {metrics['extra_px']}px  |  "
        f"Under-seg: {metrics['mask_under_px']}px  |  "
        f"Over-seg: {metrics['mask_over_px']}px"
    )
    draw.text((10, target_h + 30), metrics_text, fill=(180, 180, 180))
    draw.text((10, target_h + 55), f"Category: {category}", fill=(140, 140, 140))
    
    path = os.path.join(output_dir, f"edge_{idx:03d}_{category}.jpg")
    canvas.save(path, quality=95)
    return path


def main():
    parser = argparse.ArgumentParser(description="Edge failure visual diagnostic")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--output-dir", default="./edge_diagnostic")
    parser.add_argument("--backbone", default="SWIN_T")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable sampling")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    register_fashion_datasets()
    thing_classes = get_thing_classes()
    
    cfg = build_cfg(
        output_dir="./output_swin_fashion",
        backbone=args.backbone,
        num_classes=len(thing_classes),
    )
    
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.weights)
    
    # Load dataset list to enable random sampling across classes
    dataset_dicts = DatasetCatalog.get("fashion_val")
    
    # Group images by class for balanced random sampling
    from collections import defaultdict
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    class_to_imgs = defaultdict(list)
    for d in dataset_dicts:
        # Assuming one annotation per image based on previous verification
        if len(d["annotations"]) > 0:
            cat_id = d["annotations"][0]["category_id"]
            class_to_imgs[cat_id].append(d)
            
    # Sample categories then sample one image from each
    available_cats = sorted(class_to_imgs.keys())
    random.shuffle(available_cats)
    
    selected_dicts = []
    for cat_id in available_cats:
        if len(selected_dicts) >= args.num_images:
            break
        img_dict = random.choice(class_to_imgs[cat_id])
        selected_dicts.append(img_dict)

    # Use a simple mapper-based loader for the selected images
    mapper = ValidationDatasetMapper(cfg)
    
    # Aggregate failure stats
    all_metrics = []
    failure_types = {'bleeding': 0, 'missing': 0, 'balanced': 0}
    skipped_no_instances = 0
    skipped_no_detections = 0
    skipped_shape_mismatch = 0
    skipped_exceptions = 0
    
    logger.info(
        f"Generating edge diagnostics for {len(selected_dicts)} random images from random classes "
        f"(threshold={args.threshold}, seed={args.seed})..."
    )
    
    with torch.inference_mode():
        for idx, d in enumerate(selected_dicts):
            try:
                # Map the raw dict to the format the model expects
                img_data = mapper(d)

                outputs = model([img_data])
                output = outputs[0]

                if "instances" not in output or "instances" not in img_data:
                    skipped_no_instances += 1
                    logger.warning(f"  [{idx:3d}] Skipping: No instances found in output/GT")
                    continue

                pred_inst = output["instances"]
                gt_inst = img_data["instances"]

                scores = pred_inst.scores.detach().cpu()
                keep = scores > args.threshold

                # Filter predictions to classes present in GT for this image so
                # unrelated classes do not inflate apparent edge bleeding.
                if hasattr(pred_inst, "pred_classes") and len(gt_inst.gt_classes) > 0:
                    gt_classes = set(gt_inst.gt_classes.detach().cpu().tolist())
                    pred_classes = pred_inst.pred_classes.detach().cpu()
                    class_keep = torch.zeros_like(pred_classes, dtype=torch.bool)
                    for cid in gt_classes:
                        class_keep |= (pred_classes == cid)
                    keep = keep & class_keep

                if keep.sum() == 0:
                    skipped_no_detections += 1
                    max_score = scores.max().item() if scores.numel() > 0 else 0.0
                    logger.warning(
                        f"  [{idx:3d}] Skipping image: No detections > {args.threshold} "
                        f"(Max score: {max_score:.3f})"
                    )
                    continue

                pred_masks = pred_inst.pred_masks[keep].to("cuda").float()
                gt_masks = gt_inst.gt_masks.tensor.to("cuda").float()

                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    skipped_shape_mismatch += 1
                    logger.warning(
                        f"  [{idx:3d}] Skipping image: Shape mismatch "
                        f"{pred_masks.shape} vs {gt_masks.shape}"
                    )
                    continue

                # Get category name
                cat_id = gt_inst.gt_classes[0].item() if len(gt_inst.gt_classes) > 0 else -1
                cat_name = thing_classes[cat_id] if 0 <= cat_id < len(thing_classes) else "unknown"

                panels, metrics = make_edge_comparison(
                    img_data["image"], pred_masks, gt_masks, args.threshold
                )

                path = save_diagnostic(panels, metrics, idx, args.output_dir, cat_name)
                all_metrics.append(metrics)

                # Classify failure type
                if metrics['extra_px'] > metrics['missing_px'] * 2:
                    failure_types['bleeding'] += 1
                    ftype = "BLEEDING"
                elif metrics['missing_px'] > metrics['extra_px'] * 2:
                    failure_types['missing'] += 1
                    ftype = "MISSING"
                else:
                    failure_types['balanced'] += 1
                    ftype = "balanced"

                logger.info(
                    f"  [{idx:3d}] {cat_name:40s} | "
                    f"BFS={metrics['bfscore']:.3f} | "
                    f"Recall={metrics['recall']:.3f} | "
                    f"Prec={metrics['precision']:.3f} | "
                    f"{ftype}"
                )

                del outputs, pred_masks, gt_masks
            except Exception as e:
                skipped_exceptions += 1
                logger.exception(f"  [{idx:3d}] Skipping image due to exception: {e}")
    
    # Summary
    if all_metrics:
        avg_bfs = np.mean([m['bfscore'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"  EDGE FAILURE ANALYSIS SUMMARY ({len(all_metrics)} images)")
        logger.info(f"{'='*70}")
        logger.info(f"  Avg BFScore:           {avg_bfs:.4f}")
        logger.info(f"  Avg Boundary Recall:   {avg_recall:.4f}")
        logger.info(f"  Avg Boundary Precision: {avg_precision:.4f}")
        logger.info(f"")
        logger.info(f"  Failure breakdown:")
        logger.info(f"    Bleeding (over-prediction at edges): {failure_types['bleeding']}")
        logger.info(f"    Missing (under-prediction at edges): {failure_types['missing']}")
        logger.info(f"    Balanced (both):                     {failure_types['balanced']}")
        logger.info(f"")
        
        if failure_types['bleeding'] > failure_types['missing']:
            logger.info(f"  ➜ DOMINANT FAILURE: BLEEDING")
            logger.info(f"    → Masks expand beyond GT edges")
            logger.info(f"    → Fix: Stronger BCE (MASK_WEIGHT), boundary loss with precision focus")
        elif failure_types['missing'] > failure_types['bleeding']:
            logger.info(f"  ➜ DOMINANT FAILURE: MISSING")
            logger.info(f"    → Thin structures vanish, edges under-predicted")
            logger.info(f"    → Fix: Distance-weighted loss, thin-structure sampling")
        else:
            logger.info(f"  ➜ MIXED FAILURE: Both bleeding and missing")
            logger.info(f"    → Fix: Full boundary loss (Sobel supervision + distance weighting)")
        
        logger.info(f"\n  Visual diagnostics saved to: {args.output_dir}/")
        logger.info(f"  Panels: Original | GT Boundary | Pred Boundary | Edge Error | Mask Error")
        logger.info(f"  Colors: Green=correct | Blue=missing | Red=bleeding")
        logger.info(f"{'='*70}\n")
    else:
        logger.warning(f"\n{'='*70}")
        logger.warning("  EDGE FAILURE ANALYSIS: NO VALID SAMPLES")
        logger.warning(f"{'='*70}")
        logger.warning(f"  Requested images: {len(selected_dicts)}")
        logger.warning(f"  Skipped (no instances):      {skipped_no_instances}")
        logger.warning(f"  Skipped (no detections):     {skipped_no_detections}")
        logger.warning(f"  Skipped (shape mismatch):    {skipped_shape_mismatch}")
        logger.warning(f"  Skipped (exceptions):        {skipped_exceptions}")
        logger.warning("  Tips:")
        logger.warning("    - Lower threshold: --threshold 0.05")
        logger.warning("    - Verify weights/backbone pair")
        logger.warning("    - Re-run with --num-images 5 for quick debugging")
        logger.warning(f"{'='*70}\n")


if __name__ == "__main__":
    main()
