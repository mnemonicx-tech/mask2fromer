"""
config_setup.py
---------------
Builds and returns a Detectron2 CfgNode for Mask2Former instance segmentation
tuned for an RTX A4000 (16 GB VRAM) with 98 fashion classes.

Backbone choice: ResNet-50-FPN
  - Better out-of-the-box convergence than Swin-T for domain-specific data
  - Lower VRAM footprint leaves headroom to increase batch/crop size
  - Swin-T variant is provided as an easy drop-in swap (see SWIN_* variables)

Usage:
    from config_setup import build_cfg
    cfg = build_cfg(output_dir="./output", resume=False)
"""

import os
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config  # noqa: unused but kept for completeness
from mask2former import add_maskformer2_config              # mask2former package


# ---------------------------------------------------------------------------
# Paths to official Mask2Former config files shipped with the repo.
# These are relative to the mask2former project root you cloned/installed.
# ---------------------------------------------------------------------------
_M2F_ROOT = os.environ.get(
    "MASK2FORMER_ROOT",
    os.path.join(os.path.dirname(__file__), "Mask2Former"),
)

# ResNet-50 config (default)
_R50_CFG = os.path.join(
    _M2F_ROOT,
    "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
)

# Swin-T config (swap-in if you want to try Swin)
_SWINT_CFG = os.path.join(
    _M2F_ROOT,
    "configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
)


def build_cfg(
    output_dir: str = "./output",
    resume: bool = False,
    backbone: str = "R50",          # "R50" | "SWIN_T"
    num_classes: int = 98,
    train_dataset: str = "fashion_train",
    val_dataset: str = "fashion_val",
) -> "CfgNode":
    """
    Return a fully configured CfgNode ready for training.

    Parameters
    ----------
    output_dir  : Where to write checkpoints and logs.
    resume      : If True, resume from last checkpoint in output_dir.
    backbone    : "R50" for ResNet-50, "SWIN_T" for Swin-Tiny.
    num_classes : Number of foreground classes (98 for this project).
    train_dataset / val_dataset : Registered dataset names.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # -----------------------------------------------------------------------
    # Base config from YAML
    # -----------------------------------------------------------------------
    base_cfg = _R50_CFG if backbone.upper() == "R50" else _SWINT_CFG
    cfg.merge_from_file(base_cfg)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST  = (val_dataset,)

    # -----------------------------------------------------------------------
    # Model head — set num classes correctly
    # -----------------------------------------------------------------------
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    # Mask2Former uses a panoptic/instance head; the number of queries is tuned
    # for COCO (133 classes, 100 queries).  For 98 classes keep 100 queries.
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    # -----------------------------------------------------------------------
    # Backbone pretrained weights (downloaded automatically by Detectron2)
    # -----------------------------------------------------------------------
    if backbone.upper() == "R50":
        cfg.MODEL.WEIGHTS = (
            "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
        )
    else:
        # Swin-T ImageNet-1K weights
        cfg.MODEL.WEIGHTS = (
            "https://github.com/SwinTransformer/storage/releases/download/"
            "v1.0.0/swin_tiny_patch4_window7_224.pth"
        )

    # -----------------------------------------------------------------------
    # Input / augmentation
    # -----------------------------------------------------------------------
    # A100-80GB: push resolution high for sharp garment edges & fine seams.
    # Multi-scale training with bias toward larger sizes for detail.
    cfg.INPUT.MIN_SIZE_TRAIN       = (800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280)
    cfg.INPUT.MAX_SIZE_TRAIN       = 1536
    cfg.INPUT.MIN_SIZE_TEST        = 1024
    cfg.INPUT.MAX_SIZE_TEST        = 1536
    cfg.INPUT.FORMAT               = "RGB"
    cfg.INPUT.RANDOM_FLIP          = "horizontal"

    # -----------------------------------------------------------------------
    # Solver — tuned for A100-80GB PCIe
    # -----------------------------------------------------------------------
    # A100-80GB can fit 8 images/iter at full resolution (1024).
    # No grad accumulation needed → effective batch = 8.
    # Scale LR: base 1e-4 @ bs16 → 5e-5 @ bs8.
    cfg.SOLVER.IMS_PER_BATCH           = 8

    _EFFECTIVE_ITERS = 450_000
    cfg.SOLVER.MAX_ITER                = _EFFECTIVE_ITERS

    cfg.SOLVER.BASE_LR                 = 5e-5
    cfg.SOLVER.WEIGHT_DECAY            = 0.05

    # AdamW (Mask2Former default)
    cfg.SOLVER.OPTIMIZER               = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER     = 0.1   # lower LR for pretrained backbone

    # Cosine decay with linear warmup
    cfg.SOLVER.LR_SCHEDULER_NAME       = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_FACTOR           = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS            = 1000
    cfg.SOLVER.WARMUP_METHOD           = "linear"

    # Clip gradients to prevent spikes on fashion's long-tail distribution
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED       = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE     = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE    = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE     = 2.0

    # -----------------------------------------------------------------------
    # Mixed precision (AMP / FP16)
    # -----------------------------------------------------------------------
    cfg.SOLVER.AMP.ENABLED = True

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------
    cfg.SOLVER.CHECKPOINT_PERIOD = 5_000    # save every 5 k iterations
    cfg.TEST.EVAL_PERIOD          = 5_000   # eval every 5 k iterations

    # -----------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------
    # A100 machine has 28 CPUs and 120GB RAM — use more workers for throughput.
    cfg.DATALOADER.NUM_WORKERS           = 8
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from register_dataset import register_fashion_datasets
    register_fashion_datasets()

    cfg = build_cfg(output_dir="./output_test")
    print(cfg)
    print("\nConfig built successfully.")
    print(f"  Backbone        : {cfg.MODEL.BACKBONE.NAME}")
    print(f"  Num classes     : {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  Max iterations  : {cfg.SOLVER.MAX_ITER}")
    print(f"  AMP enabled     : {cfg.SOLVER.AMP.ENABLED}")
    print(f"  IMS_PER_BATCH   : {cfg.SOLVER.IMS_PER_BATCH}")
