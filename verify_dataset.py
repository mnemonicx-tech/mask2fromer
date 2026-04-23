"""
verify_dataset.py — Deep dataset health check with visual samples.
Catches: broken images, empty masks, corrupt polygons, class imbalance, mask quality issues.

Usage:
    python verify_dataset.py                           # check both splits
    python verify_dataset.py --split val --samples 10  # check val, save 10 visual samples
"""

import os
import json
import argparse
import random
import numpy as np
from collections import Counter, defaultdict
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_coco_json(json_path):
    logger.info(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    logger.info(f"  Images: {len(data['images'])}")
    logger.info(f"  Annotations: {len(data['annotations'])}")
    logger.info(f"  Categories: {len(data['categories'])}")
    return data


def check_images(data, image_dir, max_check=500):
    """Verify image files exist and are readable."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  IMAGE HEALTH CHECK (sampling {max_check})")
    logger.info(f"{'='*60}")
    
    images = data["images"]
    sample = random.sample(images, min(max_check, len(images)))
    
    missing = []
    corrupt = []
    sizes = []
    
    for img in sample:
        path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(path):
            # Try just the basename
            path = os.path.join(image_dir, os.path.basename(img["file_name"]))
        
        if not os.path.exists(path):
            missing.append(img["file_name"])
            continue
        
        try:
            with Image.open(path) as pil_img:
                pil_img.verify()
            sizes.append(os.path.getsize(path))
        except Exception as e:
            corrupt.append((img["file_name"], str(e)))
    
    checked = len(sample)
    logger.info(f"  Checked:  {checked}")
    logger.info(f"  Missing:  {len(missing)} {'⚠️' if missing else '✅'}")
    logger.info(f"  Corrupt:  {len(corrupt)} {'⚠️' if corrupt else '✅'}")
    
    if missing:
        logger.info(f"\n  First 5 missing:")
        for m in missing[:5]:
            logger.info(f"    ✗ {m}")
    
    if corrupt:
        logger.info(f"\n  First 5 corrupt:")
        for f, e in corrupt[:5]:
            logger.info(f"    ✗ {f}: {e}")
    
    if sizes:
        sizes = np.array(sizes)
        logger.info(f"\n  File sizes: min={sizes.min()/1024:.0f}KB  avg={sizes.mean()/1024:.0f}KB  max={sizes.max()/1024:.0f}KB")
        tiny = (sizes < 1024).sum()
        if tiny > 0:
            logger.info(f"  ⚠️ {tiny} files < 1KB (likely corrupt)")
    
    return missing, corrupt


def check_annotations(data):
    """Verify annotation quality: masks, areas, categories."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  ANNOTATION HEALTH CHECK")
    logger.info(f"{'='*60}")
    
    anns = data["annotations"]
    cats = {c["id"]: c["name"] for c in data["categories"]}
    img_lookup = {img["id"]: img for img in data["images"]}
    
    # Build image → annotations map
    img_to_anns = defaultdict(list)
    for ann in anns:
        img_to_anns[ann["image_id"]].append(ann)
    
    issues = {
        "no_segmentation": [],
        "empty_segmentation": [],
        "zero_area": [],
        "tiny_area": [],
        "huge_area": [],
        "unknown_category": [],
        "missing_image_id": [],
        "negative_bbox": [],
    }
    
    cat_counts = Counter()
    anns_per_image = []
    area_ratios = []
    
    for ann in anns:
        cat_id = ann.get("category_id")
        img_id = ann.get("image_id")
        
        # Category check
        if cat_id not in cats:
            issues["unknown_category"].append(ann["id"])
        else:
            cat_counts[cats[cat_id]] += 1
        
        # Image reference check
        if img_id not in img_lookup:
            issues["missing_image_id"].append(ann["id"])
            continue
        
        img_info = img_lookup[img_id]
        img_area = img_info["height"] * img_info["width"]
        
        # Segmentation check
        seg = ann.get("segmentation")
        if seg is None:
            issues["no_segmentation"].append(ann["id"])
        elif isinstance(seg, list):
            # Polygon format
            if len(seg) == 0 or all(len(p) == 0 for p in seg):
                issues["empty_segmentation"].append(ann["id"])
        elif isinstance(seg, dict):
            # RLE format
            if seg.get("counts") is None:
                issues["empty_segmentation"].append(ann["id"])
        
        # Area check
        area = ann.get("area", 0)
        if area == 0:
            issues["zero_area"].append(ann["id"])
        elif area < 100:
            issues["tiny_area"].append(ann["id"])
        elif img_area > 0 and area > img_area * 0.95:
            issues["huge_area"].append(ann["id"])
        
        if img_area > 0:
            area_ratios.append(area / img_area)
        
        # BBox check
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            if bbox[2] <= 0 or bbox[3] <= 0:
                issues["negative_bbox"].append(ann["id"])
    
    # Annotations per image
    for img in data["images"]:
        count = len(img_to_anns.get(img["id"], []))
        anns_per_image.append(count)
    
    anns_per_image = np.array(anns_per_image)
    
    logger.info(f"\n  Total annotations: {len(anns)}")
    logger.info(f"  Total images: {len(data['images'])}")
    logger.info(f"  Annotations/image: min={anns_per_image.min()} avg={anns_per_image.mean():.1f} max={anns_per_image.max()}")
    
    zero_ann_images = (anns_per_image == 0).sum()
    single_ann_images = (anns_per_image == 1).sum()
    multi_ann_images = (anns_per_image > 1).sum()
    
    logger.info(f"\n  Images with 0 annotations: {zero_ann_images} {'⚠️' if zero_ann_images > 0 else '✅'}")
    logger.info(f"  Images with 1 annotation:  {single_ann_images}")
    logger.info(f"  Images with 2+ annotations: {multi_ann_images}")
    
    logger.info(f"\n  --- Issue Summary ---")
    for issue_name, ids in issues.items():
        status = "⚠️" if ids else "✅"
        logger.info(f"  {issue_name}: {len(ids)} {status}")
    
    # Area distribution
    if area_ratios:
        area_ratios = np.array(area_ratios)
        logger.info(f"\n  --- Mask Area (as % of image) ---")
        logger.info(f"  min:    {area_ratios.min()*100:.2f}%")
        logger.info(f"  median: {np.median(area_ratios)*100:.2f}%")
        logger.info(f"  mean:   {area_ratios.mean()*100:.2f}%")
        logger.info(f"  max:    {area_ratios.max()*100:.2f}%")
        logger.info(f"  < 1%:   {(area_ratios < 0.01).sum()} annotations")
        logger.info(f"  < 5%:   {(area_ratios < 0.05).sum()} annotations")
        logger.info(f"  > 50%:  {(area_ratios > 0.50).sum()} annotations")
    
    # Category distribution
    logger.info(f"\n  --- Category Distribution (top 10 / bottom 10) ---")
    sorted_cats = cat_counts.most_common()
    logger.info(f"  TOP 10:")
    for name, count in sorted_cats[:10]:
        logger.info(f"    {count:>6}  {name}")
    logger.info(f"  BOTTOM 10:")
    for name, count in sorted_cats[-10:]:
        logger.info(f"    {count:>6}  {name}")
    
    # Imbalance ratio
    if sorted_cats:
        max_count = sorted_cats[0][1]
        min_count = sorted_cats[-1][1]
        ratio = max_count / max(min_count, 1)
        logger.info(f"\n  Imbalance ratio (max/min): {ratio:.1f}x {'⚠️ severe' if ratio > 20 else '✅'}")
    
    return issues, cat_counts


def render_samples(data, image_dir, output_dir, num_samples=5):
    """Render GT masks overlaid on images for visual inspection."""
    try:
        from pycocotools.coco import COCO
        import tempfile
    except ImportError:
        logger.info("  ⚠️ pycocotools not installed, skipping visual samples")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO API needs a file path
    coco = COCO.__new__(COCO)
    coco.dataset = data
    coco.createIndex()
    
    img_ids = random.sample(list(coco.imgs.keys()), min(num_samples, len(coco.imgs)))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  VISUAL SAMPLES → {output_dir}")
    logger.info(f"{'='*60}")
    
    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, os.path.basename(img_info["file_name"]))
        
        if not os.path.exists(img_path):
            logger.info(f"  ⚠️ Missing: {img_info['file_name']}")
            continue
        
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # Create colored mask overlay
            overlay = img.copy().astype(np.float32)
            
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
            ]
            
            mask_count = 0
            for i, ann in enumerate(anns):
                try:
                    mask = coco.annToMask(ann)
                    color = colors[i % len(colors)]
                    for c in range(3):
                        overlay[:, :, c] = np.where(mask > 0, overlay[:, :, c] * 0.5 + color[c] * 0.5, overlay[:, :, c])
                    mask_count += 1
                except Exception as e:
                    logger.info(f"  ⚠️ Failed to decode mask for ann {ann['id']}: {e}")
            
            out_path = os.path.join(output_dir, f"sample_{img_id}_{mask_count}masks.jpg")
            Image.fromarray(overlay.astype(np.uint8)).save(out_path)
            
            cat_names = [coco.cats[a["category_id"]]["name"] for a in anns if a["category_id"] in coco.cats]
            logger.info(f"  ✅ {os.path.basename(out_path)} — {mask_count} masks — categories: {cat_names}")
            
        except Exception as e:
            logger.info(f"  ⚠️ Failed: {img_info['file_name']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dataset health check")
    parser.add_argument("--train-json", default="/ephemeral/training_data/annotations/instances_train.json")
    parser.add_argument("--val-json", default="/ephemeral/training_data/annotations/instances_val.json")
    parser.add_argument("--train-images", default="/ephemeral/training_data/images/train")
    parser.add_argument("--val-images", default="/ephemeral/training_data/images/val")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--samples", type=int, default=5, help="Number of visual samples to render")
    parser.add_argument("--output-dir", default="./dataset_check")
    args = parser.parse_args()

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", args.train_json, args.train_images))
    if args.split in ("val", "both"):
        splits.append(("val", args.val_json, args.val_images))

    for split_name, json_path, image_dir in splits:
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"  CHECKING: {split_name.upper()} SPLIT")
        logger.info(f"{'#'*60}")
        
        data = load_coco_json(json_path)
        check_images(data, image_dir)
        check_annotations(data)
        render_samples(data, image_dir, os.path.join(args.output_dir, split_name), num_samples=args.samples)
    
    logger.info(f"\n\n{'='*60}")
    logger.info(f"  DONE — visual samples saved to {args.output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
