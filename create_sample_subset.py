"""
create_sample_subset.py
-----------------------
Memory-efficient COCO subset creator for large annotation files.

Uses ijson for **streaming** JSON parsing so the full file is never loaded
into RAM — critical when annotation files are 10+ GB and the machine has
limited memory (e.g., 21 GB).

It filters categories by name, remaps category ids to contiguous [1..N],
and samples a capped number of images per split.

Output layout:
  <output-root>/
    annotations/instances_train.json
    annotations/instances_val.json
    classes.txt

The output JSON keeps original file_name values, so you can reuse the original
image directories without copying images.

Example:
python create_sample_subset.py \
  --data-root /mnt/large_volume/training_data \
  --output-root /mnt/large_volume/training_data_sample6 \
  --classes shirt,t-shirt,jeans,jacket,dress,skirt \
  --max-train-images 12000 \
  --max-val-images 2000
"""

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from decimal import Decimal
from typing import Dict, List, Tuple


def _ensure_ijson():
    """Import ijson, auto-installing if missing."""
    try:
        import ijson
        return ijson
    except ImportError:
        print("[subset] ijson not installed — installing now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ijson"],
            stdout=subprocess.DEVNULL,
        )
        import ijson
        return ijson


class _DecimalEncoder(json.JSONEncoder):
    """ijson parses numbers as Decimal — convert back to int/float for json.dump."""
    def default(self, o):
        if isinstance(o, Decimal):
            return int(o) if o == int(o) else float(o)
        return super().default(o)


def _save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=_DecimalEncoder)


def _extract_top_level_metadata(ijson, json_path: str) -> Tuple[Dict, List[Dict]]:
    """Read lightweight top-level COCO metadata without loading full JSON."""
    info = {}
    licenses = []

    with open(json_path, "rb") as f:
        info = next(ijson.items(f, "info"), {})

    with open(json_path, "rb") as f:
        licenses = next(ijson.items(f, "licenses"), [])

    return info, licenses


def _build_subset_streaming(
    ijson,
    json_path: str,
    selected_class_names: List[str],
    max_images: int,
    seed: int,
) -> Tuple[Dict, Dict[str, int]]:
    """
    Build a COCO subset by streaming through the JSON file in 3 passes.
    Peak memory ≈ size of the *filtered* subset, not the whole file.
    """
    basename = os.path.basename(json_path)
    info, licenses = _extract_top_level_metadata(ijson, json_path)

    # ------------------------------------------------------------------
    # Pass 1: categories (tiny — always fits in memory)
    # ------------------------------------------------------------------
    print(f"  [stream] {basename} — Pass 1/3: reading categories...")
    categories = []
    with open(json_path, "rb") as f:
        for cat in ijson.items(f, "categories.item"):
            categories.append(cat)

    name_to_cat = {c["name"]: c for c in categories}
    missing = [c for c in selected_class_names if c not in name_to_cat]
    if missing:
        raise ValueError(f"Classes not found in categories: {missing}")

    selected_old_ids = set(name_to_cat[n]["id"] for n in selected_class_names)
    old_to_new = {
        name_to_cat[n]["id"]: i + 1
        for i, n in enumerate(selected_class_names)
    }

    # ------------------------------------------------------------------
    # Pass 2: stream annotations — keep only matching category_id
    # ------------------------------------------------------------------
    print(f"  [stream] {basename} — Pass 2/3: filtering annotations "
          f"for {len(selected_class_names)} classes...")
    filtered_anns = []
    image_ids = set()
    scanned = 0
    with open(json_path, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            scanned += 1
            if scanned % 2_000_000 == 0:
                print(f"  [stream]   ...scanned {scanned:,} annotations")
            if ann.get("category_id") in selected_old_ids:
                image_ids.add(ann["image_id"])
                filtered_anns.append(ann)

    print(f"  [stream]   scanned {scanned:,} total, "
          f"kept {len(filtered_anns):,} annotations")
    print(f"  [stream]   images with target classes: {len(image_ids):,}")

    if not filtered_anns:
        raise ValueError("No annotations left after class filtering.")

    # Sample images if over budget
    image_ids_list = sorted(image_ids)
    if 0 < max_images < len(image_ids_list):
        rnd = random.Random(seed)
        image_ids_list = rnd.sample(image_ids_list, max_images)
        print(f"  [stream]   sampled down to {max_images} images")

    keep_ids = set(image_ids_list)
    del image_ids, image_ids_list

    # Trim annotations to sampled image set & remap category ids
    filtered_anns = [a for a in filtered_anns if a["image_id"] in keep_ids]
    for a in filtered_anns:
        a["category_id"] = old_to_new[a["category_id"]]

    # ------------------------------------------------------------------
    # Pass 3: stream images — keep only those in keep_ids
    # ------------------------------------------------------------------
    print(f"  [stream] {basename} — Pass 3/3: filtering images...")
    filtered_images = []
    with open(json_path, "rb") as f:
        for img in ijson.items(f, "images.item"):
            if img.get("id") in keep_ids:
                filtered_images.append(img)
                if len(filtered_images) == len(keep_ids):
                    break  # all found — stop early

    # Build new categories
    new_categories = [
        {
            "id": i + 1,
            "name": n,
            "supercategory": name_to_cat[n].get("supercategory", "fashion"),
        }
        for i, n in enumerate(selected_class_names)
    ]

    out = {
        "info": info,
        "licenses": licenses,
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": new_categories,
    }
    stats = {
        "num_images": len(filtered_images),
        "num_annotations": len(filtered_anns),
        "num_categories": len(new_categories),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sampled COCO subset (memory-efficient streaming)"
    )
    parser.add_argument("--data-root", required=True, help="Original dataset root")
    parser.add_argument("--output-root", required=True, help="Output subset root")
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated class names, e.g. shirt,t-shirt,jeans,jacket,dress,skirt",
    )
    parser.add_argument("--max-train-images", type=int, default=12000)
    parser.add_argument("--max-val-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(selected_classes) < 2:
        raise ValueError("Please select at least 2 classes.")

    ijson = _ensure_ijson()

    train_json = os.path.join(args.data_root, "annotations", "instances_train.json")
    val_json = os.path.join(args.data_root, "annotations", "instances_val.json")

    out_ann_dir = os.path.join(args.output_root, "annotations")
    out_train_json = os.path.join(out_ann_dir, "instances_train.json")
    out_val_json = os.path.join(out_ann_dir, "instances_val.json")

    # --- Process train split ---
    print(f"[subset] Processing train split...")
    train_subset, train_stats = _build_subset_streaming(
        ijson, train_json,
        selected_class_names=selected_classes,
        max_images=args.max_train_images,
        seed=args.seed,
    )
    _save_json(out_train_json, train_subset)
    del train_subset  # free memory before val split
    gc.collect()

    # --- Process val split ---
    print(f"[subset] Processing val split...")
    val_subset, val_stats = _build_subset_streaming(
        ijson, val_json,
        selected_class_names=selected_classes,
        max_images=args.max_val_images,
        seed=args.seed + 1,
    )
    _save_json(out_val_json, val_subset)
    del val_subset
    gc.collect()

    classes_file = os.path.join(args.output_root, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_classes) + "\n")

    print("[subset] done")
    print(f"[subset] classes_file={classes_file}")
    print(f"[subset] train_json={out_train_json} stats={train_stats}")
    print(f"[subset] val_json={out_val_json} stats={val_stats}")
    print("[subset] Use original image dirs:")
    print(f"[subset] train_images={os.path.join(args.data_root, 'images', 'train')}")
    print(f"[subset] val_images={os.path.join(args.data_root, 'images', 'val')}")


if __name__ == "__main__":
    main()
