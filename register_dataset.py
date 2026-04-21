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
import pickle
import random
from typing import Dict, List

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# ---------------------------------------------------------------------------
# Fashion class names — must match classes.txt / COCO JSON category order exactly.
# Source of truth: /ephemeral/training_data/classes.txt (97 classes)
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

assert len(FASHION_CLASSES) == 97, (
    f"FASHION_CLASSES must have exactly 97 entries, got {len(FASHION_CLASSES)}. "
    "Check for duplicates or missing classes."
)


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
    data_root = os.environ.get("FASHION_DATA_ROOT", "/ephemeral/training_data")

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

    datasets = {
        "fashion_train": {"json": train_json, "images": train_images},
        "fashion_val":   {"json": val_json,   "images": val_images},
    }
    
    mini_json = os.path.join(data_root, "annotations", "instances_val_mini.json")
    if os.path.exists(mini_json):
        datasets["fashion_val_mini"] = {
            "json": mini_json,
            "images": val_images
        }
        
    return datasets


def register_fashion_datasets() -> None:
    """
    Register all fashion splits.  Safe to call multiple times — skips
    splits that are already registered.
    """
    datasets = get_datasets()
    classes  = get_thing_classes()

    for name, paths in datasets.items():
        if name in DatasetCatalog:
            continue  # already registered

        register_coco_instances(
            name=name,
            metadata={},
            json_file=paths["json"],
            image_root=paths["images"],
        )

        meta = MetadataCatalog.get(name)
        meta.thing_classes = classes
        # thing_dataset_id_to_contiguous_id is built automatically by
        # register_coco_instances from the JSON — no manual override needed
        # unless your category IDs are non-contiguous.

        print(
            f"[register_dataset] Registered '{name}' — "
            f"{len(classes)} classes | json={paths['json']}"
        )


def _load_coco_cached(json_file: str) -> "COCO":
    """
    Load a COCO object from a pickle cache when available.

    Cache file: <json_file>.coco_cache.pkl
    Invalidation: cache is discarded whenever the JSON is newer than the cache.

    On a 120 GB RAM machine this reduces repeated annotation loads from
    ~120 s (JSON parse) to ~3–5 s (pickle deserialise).
    """
    cache_path = json_file + ".coco_cache.pkl"
    try:
        json_mtime  = os.path.getmtime(json_file)
        cache_mtime = os.path.getmtime(cache_path) if os.path.isfile(cache_path) else 0
        if cache_mtime >= json_mtime:
            import logging
            logging.getLogger("register_dataset").info(
                "[coco_cache] Loading from cache: %s", cache_path
            )
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass  # cache miss or unreadable — fall through to full load

    coco = COCO(json_file)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(coco, f, protocol=pickle.HIGHEST_PROTOCOL)
        import logging
        logging.getLogger("register_dataset").info(
            "[coco_cache] Saved cache: %s", cache_path
        )
    except Exception:
        pass  # cache write failure is non-fatal
    return coco


def _ram_aware_sample_size() -> int:
    """
    Return a sensible annotation sample size based on available RAM.

    Available RAM   Sample size
    ──────────────  ──────────────────────────────────
    ≥ 60 GB         full scan  (0 → validated below)
    ≥ 20 GB         100 000
    ≥ 8 GB           30 000
    <  8 GB          10 000  (safe default)
    """
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        if avail_gb >= 60:
            return 0          # signals full scan
        if avail_gb >= 20:
            return 100_000
        if avail_gb >= 8:
            return 30_000
    except ImportError:
        pass
    return 10_000


def get_json_classes(json_file: str) -> List[str]:
    """Load class names from COCO JSON ordered by category id (uses cache)."""
    coco = _load_coco_cached(json_file)
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
            f"  idx={i}: json='{j}' vs thing_classes='{t}'"
            for i, j, t in mismatches[:10]
        )
        raise ValueError(
            "thing_classes mismatch with COCO JSON. "
            "Fix class names/order before training.\n"
            f"First mismatches:\n{preview}"
        )


def validate_coco_annotations(
    json_file: str,
    sample_size: int = -1,
    full_scan: bool = False,
) -> Dict[str, int]:
    """
    COCO annotation sanity checks.

    sample_size=-1 (default) means auto-detect based on available RAM via
    _ram_aware_sample_size(): full scan on ≥60 GB, scaled down otherwise.

    FIX #2: Changed from sequential first-N sample to RANDOM sample so
            corrupt annotations later in the 247K dataset are detected.
    FIX #3: Added zero-area, short-polygon, and RLE decode checks that
            were previously missing.

    Parameters
    ----------
    json_file   : Path to COCO annotations JSON.
    sample_size : Annotations to check. -1 = auto (RAM-aware). 0 = all.
    full_scan   : If True, scan ALL annotations regardless of sample_size.
    """
    coco    = _load_coco_cached(json_file)
    img_ids = set(coco.getImgIds())
    cat_ids = set(coco.getCatIds())
    ann_ids = list(coco.getAnnIds())

    if len(ann_ids) == 0:
        raise ValueError(f"No annotations found in {json_file}")

    if full_scan:
        sampled_ids = ann_ids
    else:
        n = _ram_aware_sample_size() if sample_size == -1 else sample_size
        sampled_ids = ann_ids if n == 0 else random.sample(ann_ids, min(n, len(ann_ids)))
    anns = coco.loadAnns(sampled_ids)

    missing_seg = 0
    bad_cat = 0
    bad_img = 0
    bad_rle = 0
    zero_area = 0
    short_poly = 0
    bad_examples: List[str] = []

    for ann in anns:
        ann_id = ann.get("id", "?")
        img_id = ann.get("image_id", "?")

        # 1. Segmentation present
        seg = ann.get("segmentation", None)
        if seg is None or (isinstance(seg, list) and len(seg) == 0):
            missing_seg += 1
            bad_examples.append(f"ann_id={ann_id} img_id={img_id}: missing segmentation")
            continue

        # 2. Polygon checks
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) < 6:          # fewer than 3 (x,y) pairs
                    short_poly += 1
                    bad_examples.append(
                        f"ann_id={ann_id} img_id={img_id}: polygon too short ({len(poly)} coords)"
                    )

        # 3. RLE decode check
        elif isinstance(seg, dict):
            try:
                m = maskUtils.decode(seg)
                if m.sum() == 0:
                    bad_rle += 1
                    bad_examples.append(f"ann_id={ann_id} img_id={img_id}: empty RLE mask")
            except Exception as e:
                bad_rle += 1
                bad_examples.append(f"ann_id={ann_id} img_id={img_id}: RLE decode error — {e}")

        # 4. Zero / negative area
        if ann.get("area", 1) <= 0:
            zero_area += 1
            bad_examples.append(f"ann_id={ann_id} img_id={img_id}: zero/negative area={ann.get('area')}")

        # 5. Category ID in range
        if ann.get("category_id") not in cat_ids:
            bad_cat += 1
            bad_examples.append(
                f"ann_id={ann_id} img_id={img_id}: invalid category_id={ann.get('category_id')}"
            )

        # 6. Image ID exists
        if ann.get("image_id") not in img_ids:
            bad_img += 1
            bad_examples.append(f"ann_id={ann_id}: image_id={img_id} not in dataset")

    total_bad = missing_seg + bad_cat + bad_img + bad_rle + zero_area + short_poly
    if bad_examples:
        import logging
        _log = logging.getLogger("register_dataset")
        _log.warning("Annotation issues found (%d total):", total_bad)
        for ex in bad_examples[:20]:
            _log.warning("  %s", ex)

    if missing_seg or bad_cat or bad_img or bad_rle or zero_area or short_poly:
        raise ValueError(
            "COCO annotation validation failed: "
            f"missing_seg={missing_seg}, bad_cat={bad_cat}, bad_img={bad_img}, "
            f"bad_rle={bad_rle}, zero_area={zero_area}, short_poly={short_poly}"
        )

    return {
        "num_images":          len(img_ids),
        "num_categories":      len(cat_ids),
        "num_annotations":     len(ann_ids),
        "checked_annotations": len(sampled_ids),
    }


def run_preflight_checks(full_scan: bool = False) -> None:
    """Fail-fast validation for class names and annotation schema."""
    datasets = get_datasets()
    classes  = get_thing_classes()

    for split_name, paths in datasets.items():
        json_file = paths["json"]
        validate_thing_classes_against_json(json_file, classes)
        stats = validate_coco_annotations(json_file, full_scan=full_scan)
        print(
            f"[preflight] {split_name}: "
            f"images={stats['num_images']} categories={stats['num_categories']} "
            f"annotations={stats['num_annotations']} checked={stats['checked_annotations']}"
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-scan", action="store_true", help="Scan ALL annotations (slow)")
    args = parser.parse_args()

    run_preflight_checks(full_scan=args.full_scan)
    register_fashion_datasets()

    dataset_dicts = DatasetCatalog.get("fashion_train")
    print(f"Train samples : {len(dataset_dicts)}")
    print(f"Sample entry  : {list(dataset_dicts[0].keys())}")

    meta = MetadataCatalog.get("fashion_train")
    print(f"Classes       : {meta.thing_classes[:5]} … (total {len(meta.thing_classes)})")