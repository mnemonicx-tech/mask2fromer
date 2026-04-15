"""
register_dataset.py
-------------------
Registers COCO-format train/val datasets for Mask2Former instance segmentation.
Supports 98 fashion categories with proper metadata.

Usage:
    from register_dataset import register_fashion_datasets
    register_fashion_datasets()
"""

import os
from typing import Dict, List

from pycocotools.coco import COCO
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# ---------------------------------------------------------------------------
# Fashion class names expected in COCO JSON category "name".
# Keep order aligned to dataset category-id ordering.
# ---------------------------------------------------------------------------
FASHION_CLASSES = [
    "bottomwear_men_cargo_pants",
    "bottomwear_men_chinos",
    "bottomwear_men_dhoti",
    "bottomwear_men_formal_trousers",
    "bottomwear_men_jeans",
    "bottomwear_men_joggers",
    "bottomwear_men_relaxed_jeans",
    "bottomwear_men_shorts",
    "bottomwear_men_skinny_jeans",
    "bottomwear_men_slim_jeans",
    "bottomwear_men_track_pants",
    "bottomwear_women_bootcut_jeans",
    "bottomwear_women_cargos",
    "bottomwear_women_jeans",
    "bottomwear_women_joggers",
    "bottomwear_women_leggings",
    "bottomwear_women_midi_skirt",
    "bottomwear_women_mini_skirt",
    "bottomwear_women_mom_fit_jeans",
    "bottomwear_women_shorts",
    "bottomwear_women_skinny_jeans",
    "bottomwear_women_skirt",
    "bottomwear_women_trousers",
    "bottomwear_women_wide_leg_jeans",
    "ethnic_wear_men_long_kurta",
    "ethnic_wear_men_nehru_jacket",
    "ethnic_wear_men_sherwani",
    "ethnic_wear_men_short_kurta",
    "ethnic_wear_men_waistcoat",
    "ethnic_wear_women_anarkali",
    "ethnic_wear_women_dupatta",
    "ethnic_wear_women_kurti",
    "ethnic_wear_women_lehenga_choli",
    "ethnic_wear_women_palazzo_set",
    "ethnic_wear_women_salwar_suit",
    "ethnic_wear_women_saree",
    "ethnic_wear_women_sharara",
    "fusion_wear_women_indo_western_dress",
    "fusion_wear_women_kaftan",
    "general_unisex_hoodie",
    "general_unisex_joggers",
    "general_unisex_oversized_t_shirt",
    "general_unisex_raincoat",
    "general_unisex_sweatshirt",
    "general_unisex_tracksuit",
    "sleepwear_men_lounge_pants",
    "sleepwear_men_pyjama_set",
    "sleepwear_women_night_suit",
    "sleepwear_women_nightgown",
    "sleepwear_women_robes",
    "sportswear_men_gym_shorts",
    "sportswear_men_sports_t_shirt",
    "sportswear_men_tracksuit",
    "sportswear_women_gym_leggings",
    "sportswear_women_gym_top",
    "sportswear_women_sports_bra",
    "topwear_men_blazer",
    "topwear_men_bomber_jacket",
    "topwear_men_casual_shirt",
    "topwear_men_coat",
    "topwear_men_denim_jacket",
    "topwear_men_denim_shirt",
    "topwear_men_formal_shirt",
    "topwear_men_graphic_t_shirt",
    "topwear_men_henley_t_shirt",
    "topwear_men_hoodie",
    "topwear_men_leather_jacket",
    "topwear_men_linen_shirt",
    "topwear_men_oversized_t_shirt",
    "topwear_men_polo_t_shirt",
    "topwear_men_printed_shirt",
    "topwear_men_puffer_jacket",
    "topwear_men_sweater",
    "topwear_men_sweatshirt",
    "topwear_men_t_shirt",
    "topwear_men_windcheater",
    "topwear_women_blouse",
    "topwear_women_bodysuit",
    "topwear_women_bomber_jacket",
    "topwear_women_camisole",
    "topwear_women_crop_top",
    "topwear_women_denim_jacket",
    "topwear_women_hoodie",
    "topwear_women_leather_jacket",
    "topwear_women_shirt",
    "topwear_women_sweater",
    "topwear_women_sweatshirt",
    "topwear_women_t_shirt",
    "topwear_women_top",
    "topwear_women_trench_coat",
    "opwear_women_trench_coat",
    "tunic_nan_women",
    "western_wear_women_bodycon_dress",
    "western_wear_women_jumpsuit",
    "western_wear_women_maxi_dress",
    "western_wear_women_midi_dress",
    "western_wear_women_mini_dress",
    "western_wear_women_playsuit",
    "western_wear_women_shirt_dress",
    "western_wear_women_wrap_dress",
]

assert len(FASHION_CLASSES) > 0, "No classes configured in CLASS_NAME_MAP"

def get_thing_classes() -> List[str]:
    """
    Return class names from env file override or built-in defaults.

    Env override:
      FASHION_CLASSES_FILE=/path/to/classes.txt
      - one class name per line
      - order must match category id ordering in JSON
    """
    classes_file = os.environ.get("FASHION_CLASSES_FILE", "").strip()
    if not classes_file:
        return list(FASHION_CLASSES)

    if not os.path.isfile(classes_file):
        raise FileNotFoundError(f"FASHION_CLASSES_FILE not found: {classes_file}")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not classes:
        raise ValueError(f"FASHION_CLASSES_FILE is empty: {classes_file}")

    return classes


def get_datasets() -> Dict[str, Dict[str, str]]:
    """
    Resolve dataset paths from env (or defaults).

    Env overrides:
      FASHION_DATA_ROOT
      FASHION_TRAIN_JSON
      FASHION_VAL_JSON
      FASHION_TRAIN_IMAGES
      FASHION_VAL_IMAGES
    """
    data_root = os.environ.get("FASHION_DATA_ROOT", "/mnt/large_volume/training_data")

    train_json = os.environ.get(
        "FASHION_TRAIN_JSON",
        os.path.join(data_root, "annotations", "instances_train.json"),
    )
    val_json = os.environ.get(
        "FASHION_VAL_JSON",
        os.path.join(data_root, "annotations", "instances_val.json"),
    )
    train_images = os.environ.get(
        "FASHION_TRAIN_IMAGES",
        os.path.join(data_root, "images", "train"),
    )
    val_images = os.environ.get(
        "FASHION_VAL_IMAGES",
        os.path.join(data_root, "images", "val"),
    )

    return {
        "fashion_train": {"json": train_json, "images": train_images},
        "fashion_val": {"json": val_json, "images": val_images},
    }


def register_fashion_datasets() -> None:
    """
    Register all fashion splits.  Safe to call multiple times — skips
    splits that are already registered.
    """
    datasets = get_datasets()
    classes = get_thing_classes()

    for name, paths in datasets.items():
        if name in DatasetCatalog:
            continue  # already registered

        register_coco_instances(
            name=name,
            metadata={},           # filled in below
            json_file=paths["json"],
            image_root=paths["images"],
        )

        meta = MetadataCatalog.get(name)
        meta.thing_classes = classes
        # Detectron2 convention: thing_dataset_id_to_contiguous_id is built
        # automatically by register_coco_instances from the JSON; no need to
        # set it manually unless you have non-contiguous IDs.

        print(f"[register_dataset] Registered '{name}' — "
              f"{len(classes)} classes | "
              f"json={paths['json']}")


def get_json_classes(json_file: str) -> List[str]:
    """Load class names from COCO JSON ordered by category id."""
    coco = COCO(json_file)
    cats = coco.loadCats(coco.getCatIds())
    cats = sorted(cats, key=lambda c: c["id"])
    return [c["name"] for c in cats]


def validate_thing_classes_against_json(json_file: str, thing_classes: List[str]) -> None:
    """
    Validate class names match COCO JSON exactly.
    Raises ValueError with a helpful diff when mismatch is found.
    """
    json_classes = get_json_classes(json_file)

    if len(json_classes) != len(thing_classes):
        raise ValueError(
            "Class count mismatch: "
            f"JSON has {len(json_classes)}, thing_classes has {len(thing_classes)}"
        )

    mismatches = []
    for idx, (json_name, thing_name) in enumerate(zip(json_classes, thing_classes)):
        if json_name != thing_name:
            mismatches.append((idx, json_name, thing_name))

    if mismatches:
        preview = "\n".join(
            [
                f"  idx={i}: json='{j}' vs thing_classes='{t}'"
                for i, j, t in mismatches[:10]
            ]
        )
        raise ValueError(
            "thing_classes mismatch with COCO JSON. "
            "Fix class names/order before training.\n"
            f"First mismatches:\n{preview}"
        )


def validate_coco_annotations(json_file: str, sample_size: int = 5000) -> Dict[str, int]:
    """
    Lightweight COCO annotation sanity checks.
    - segmentation exists and is non-empty
    - category_id exists in category set
    - image_id exists in image set
    """
    coco = COCO(json_file)
    img_ids = set(coco.getImgIds())
    cat_ids = set(coco.getCatIds())
    ann_ids = coco.getAnnIds()

    if len(ann_ids) == 0:
        raise ValueError(f"No annotations found in {json_file}")

    sampled_ids = ann_ids[: min(sample_size, len(ann_ids))]
    anns = coco.loadAnns(sampled_ids)

    missing_seg = 0
    bad_cat = 0
    bad_img = 0

    for ann in anns:
        seg = ann.get("segmentation", None)
        if seg is None or (isinstance(seg, list) and len(seg) == 0):
            missing_seg += 1

        if ann.get("category_id") not in cat_ids:
            bad_cat += 1

        if ann.get("image_id") not in img_ids:
            bad_img += 1

    if missing_seg or bad_cat or bad_img:
        raise ValueError(
            "COCO annotation validation failed: "
            f"missing_segmentation={missing_seg}, bad_category_id={bad_cat}, bad_image_id={bad_img}"
        )

    return {
        "num_images": len(img_ids),
        "num_categories": len(cat_ids),
        "num_annotations": len(ann_ids),
        "checked_annotations": len(sampled_ids),
    }


def run_preflight_checks() -> None:
    """Fail-fast validation for class names and annotation schema."""
    datasets = get_datasets()
    classes = get_thing_classes()

    for split_name, paths in datasets.items():
        json_file = paths["json"]
        validate_thing_classes_against_json(json_file, classes)
        stats = validate_coco_annotations(json_file)
        print(
            f"[preflight] {split_name}: "
            f"images={stats['num_images']} categories={stats['num_categories']} "
            f"annotations={stats['num_annotations']} checked={stats['checked_annotations']}"
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preflight_checks()
    register_fashion_datasets()

    from detectron2.data import build_detection_train_loader
    dataset_dicts = DatasetCatalog.get("fashion_train")
    print(f"Train samples : {len(dataset_dicts)}")
    print(f"Sample entry  : {list(dataset_dicts[0].keys())}")

    meta = MetadataCatalog.get("fashion_train")
    print(f"Classes       : {meta.thing_classes[:5]} ... (total {len(meta.thing_classes)})")
