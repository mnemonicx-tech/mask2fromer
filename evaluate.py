"""
evaluate.py
-----------
Quick checkpoint evaluator using the stratified 200-image validation subset.
Reports edge-focused metrics (BFScore, Boundary IoU) for checkpoint selection.

Usage:
    python evaluate.py --weights output_swin_fashion/model_0042499.pth
    python evaluate.py --weights model_0039999.pth model_0042499.pth model_0044999.pth
"""

import os
import sys
import argparse
import torch
import logging
from collections import defaultdict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

from config_setup import build_cfg
from register_dataset import register_fashion_datasets, get_thing_classes
from validation_utils import ValidationHook

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(cfg, model, weights_path, num_images=200):
    """Run full 200-image evaluation on a single checkpoint."""
    
    DetectionCheckpointer(model).resume_or_load(weights_path, resume=False)
    model.eval()

    with EventStorage() as storage:

        class MockTrainer:
            def __init__(self, m, s):
                self.model = m
                self.storage = s
                self.iter = 0

        trainer = MockTrainer(model, storage)
        hook = ValidationHook(cfg, "fashion_val", period=99999, num_images=num_images)
        hook.trainer = trainer
        hook.run_validation(current_iter=0)

        results = {}
        for key in [
            "val/bfscore", "val/bound_iou_strict", "val/bound_iou_loose",
            "val/fg_iou", "val/fg_dice", "val/inst_iou", "val/inst_dice",
            "val/fp_rate", "val/small_obj_bfscore", "val/small_obj_bound_iou",
        ]:
            val = storage.latest().get(key, (0.0, 0))[0]
            results[key] = val

    return results


def print_results(weights_path, results):
    """Pretty-print evaluation results for a single checkpoint."""
    name = os.path.basename(weights_path).replace(".pth", "")
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*40}")
    
    # Priority metrics first
    priority = ["val/bfscore", "val/bound_iou_strict", "val/bound_iou_loose"]
    secondary = ["val/fg_iou", "val/fg_dice", "val/inst_iou", "val/inst_dice", "val/fp_rate"]
    small = ["val/small_obj_bfscore", "val/small_obj_bound_iou"]
    
    print(f"  {'— EDGE METRICS (pick by these) —':^40}")
    for k in priority:
        print(f"  {k:<30} {results[k]:>10.4f}")
    
    print(f"  {'— REGION METRICS —':^40}")
    for k in secondary:
        print(f"  {k:<30} {results[k]:>10.4f}")
    
    print(f"  {'— SMALL OBJECT METRICS —':^40}")
    for k in small:
        print(f"  {k:<30} {results[k]:>10.4f}")
    
    print(f"{'='*60}")
    return results


def print_comparison(all_results):
    """Print side-by-side comparison table for multiple checkpoints."""
    if len(all_results) < 2:
        return
    
    names = [os.path.basename(w).replace(".pth", "") for w in all_results.keys()]
    
    print(f"\n\n{'#'*70}")
    print(f"  CHECKPOINT COMPARISON — Pick by BFScore + Bound IoU (Strict)")
    print(f"{'#'*70}\n")
    
    # Header
    header = f"  {'Metric':<28}"
    for name in names:
        header += f" {name:>14}"
    print(header)
    print(f"  {'-'*28}" + f" {'-'*14}" * len(names))
    
    priority_keys = [
        "val/bfscore", "val/bound_iou_strict", "val/bound_iou_loose",
        "val/fg_iou", "val/inst_iou", "val/fp_rate",
        "val/small_obj_bfscore",
    ]
    
    for k in priority_keys:
        row = f"  {k:<28}"
        values = [all_results[w][k] for w in all_results]
        best_val = max(values) if "fp_rate" not in k else min(values)
        
        for w in all_results:
            v = all_results[w][k]
            marker = " 🏆" if v == best_val and len(all_results) > 1 else "   "
            row += f" {v:>10.4f}{marker}"
        print(row)
    
    # Winner
    weights_list = list(all_results.keys())
    bfscores = {w: all_results[w]["val/bfscore"] for w in weights_list}
    winner = max(bfscores, key=bfscores.get)
    
    print(f"\n  ✅ RECOMMENDED: {os.path.basename(winner)}")
    print(f"     BFScore = {bfscores[winner]:.4f}")
    print(f"\n  Use for fine-tuning:")
    print(f"     python training.py --resume MODEL.WEIGHTS {winner} \\")
    print(f"       SOLVER.BASE_LR 1e-5 SOLVER.MAX_ITER 55000 \\")
    print(f"       INPUT.MIN_SCALE 0.7 INPUT.MAX_SCALE 1.3")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on stratified validation subset")
    parser.add_argument("--weights", nargs="+", required=True, help="Path(s) to checkpoint .pth files")
    parser.add_argument("--num-images", type=int, default=200, help="Number of validation images (default: 200)")
    parser.add_argument("--output-dir", type=str, default="./output_swin_fashion", help="Config output dir")
    parser.add_argument("--backbone", type=str, default="SWIN_T", help="Backbone type")
    args = parser.parse_args()

    register_fashion_datasets()
    thing_classes = get_thing_classes()

    cfg = build_cfg(
        output_dir=args.output_dir,
        resume=False,
        backbone=args.backbone,
        num_classes=len(thing_classes),
    )

    model = build_model(cfg)

    all_results = {}
    for weights_path in args.weights:
        if not os.path.exists(weights_path):
            logger.error(f"❌ File not found: {weights_path}")
            continue
            
        logger.info(f"\n🔍 Evaluating: {os.path.basename(weights_path)}")
        results = evaluate_checkpoint(cfg, model, weights_path, num_images=args.num_images)
        print_results(weights_path, results)
        all_results[weights_path] = results

    print_comparison(all_results)


if __name__ == "__main__":
    main()
