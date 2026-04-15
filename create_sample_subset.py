"""
create_sample_subset.py
-----------------------
Create a smaller COCO subset for quick training experiments (e.g., 5-6 classes).

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
import json
import os
import random
from typing import Dict, List, Tuple


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _build_subset(
    coco: Dict,
    selected_class_names: List[str],
    max_images: int,
    seed: int,
) -> Tuple[Dict, Dict[str, int]]:
    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    name_to_cat = {c["name"]: c for c in categories}
    missing = [c for c in selected_class_names if c not in name_to_cat]
    if missing:
        raise ValueError(f"Classes not found in categories: {missing}")

    selected_old_ids = [name_to_cat[name]["id"] for name in selected_class_names]
    old_to_new_cat_id = {old_id: i + 1 for i, old_id in enumerate(selected_old_ids)}

    filtered_anns = [ann for ann in annotations if ann.get("category_id") in selected_old_ids]
    if not filtered_anns:
        raise ValueError("No annotations left after class filtering.")

    image_ids_with_target = sorted({ann["image_id"] for ann in filtered_anns})
    if max_images > 0 and len(image_ids_with_target) > max_images:
        rnd = random.Random(seed)
        image_ids_with_target = rnd.sample(image_ids_with_target, max_images)

    image_id_set = set(image_ids_with_target)
    filtered_images = [img for img in images if img.get("id") in image_id_set]
    filtered_anns = [ann for ann in filtered_anns if ann.get("image_id") in image_id_set]

    for ann in filtered_anns:
        ann["category_id"] = old_to_new_cat_id[ann["category_id"]]

    new_categories = []
    for new_id, class_name in enumerate(selected_class_names, start=1):
        old_cat = name_to_cat[class_name]
        new_categories.append(
            {
                "id": new_id,
                "name": class_name,
                "supercategory": old_cat.get("supercategory", "fashion"),
            }
        )

    out = dict(coco)
    out["images"] = filtered_images
    out["annotations"] = filtered_anns
    out["categories"] = new_categories

    stats = {
        "num_images": len(filtered_images),
        "num_annotations": len(filtered_anns),
        "num_categories": len(new_categories),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sampled COCO subset for fast training")
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

    train_json = os.path.join(args.data_root, "annotations", "instances_train.json")
    val_json = os.path.join(args.data_root, "annotations", "instances_val.json")

    train_coco = _load_json(train_json)
    val_coco = _load_json(val_json)

    train_subset, train_stats = _build_subset(
        train_coco,
        selected_class_names=selected_classes,
        max_images=args.max_train_images,
        seed=args.seed,
    )
    val_subset, val_stats = _build_subset(
        val_coco,
        selected_class_names=selected_classes,
        max_images=args.max_val_images,
        seed=args.seed + 1,
    )

    out_ann_dir = os.path.join(args.output_root, "annotations")
    out_train_json = os.path.join(out_ann_dir, "instances_train.json")
    out_val_json = os.path.join(out_ann_dir, "instances_val.json")
    _save_json(out_train_json, train_subset)
    _save_json(out_val_json, val_subset)

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
