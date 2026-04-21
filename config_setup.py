"""
config_setup.py
---------------
Builds and returns a Detectron2 CfgNode for Mask2Former instance segmentation
tuned for an RTX A4000 (16 GB VRAM) with 97 fashion classes.

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
from detectron2.projects.deeplab import add_deeplab_config  # noqa: kept for completeness
from mask2former import add_maskformer2_config


# ---------------------------------------------------------------------------
# Paths to official Mask2Former config files shipped with the repo.
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
    num_classes: int = 97,
    train_dataset: str = "fashion_train",
    val_dataset: str = "fashion_val",
) -> "CfgNode":
    """
    Return a fully configured CfgNode ready for training.

    Parameters
    ----------
    output_dir       : Where to write checkpoints and logs.
    resume           : If True, resume from last checkpoint in output_dir.
    backbone         : "R50" for ResNet-50, "SWIN_T" for Swin-Tiny.
    num_classes      : Number of foreground classes (97 for this project).
    train_dataset    : Registered dataset name for training split.
    val_dataset      : Registered dataset name for validation split.
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
    # 97 classes → 200 queries gives headroom for dense multi-garment scenes
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    
    # Advanced Point Sampling for Edge Fidelity
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 20000
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.05

    # -----------------------------------------------------------------------
    # Backbone pretrained weights (downloaded automatically by Detectron2)
    # -----------------------------------------------------------------------
    if backbone.upper() == "R50":
        cfg.MODEL.WEIGHTS = (
            "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
        )
    else:
        cfg.MODEL.WEIGHTS = (
            "https://github.com/SwinTransformer/storage/releases/download/"
            "v1.0.0/swin_tiny_patch4_window7_224.pth"
        )

    # -----------------------------------------------------------------------
    # Input / augmentation
    # -----------------------------------------------------------------------
    # LSJ (Large Scale Jittering) augmentation — used by COCOInstanceNewBaselineDatasetMapper
    cfg.INPUT.IMAGE_SIZE           = 1024
    cfg.INPUT.MIN_SCALE            = 0.5
    cfg.INPUT.MAX_SCALE            = 1.5
    cfg.INPUT.DATASET_MAPPER_NAME  = "coco_instance_lsj"
    # Legacy ResizeShortestEdge keys (used by test-time inference)
    cfg.INPUT.MIN_SIZE_TRAIN       = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN       = 1333
    cfg.INPUT.MIN_SIZE_TEST        = 800
    cfg.INPUT.MAX_SIZE_TEST        = 1333
    cfg.INPUT.FORMAT               = "RGB"
    cfg.INPUT.RANDOM_FLIP          = "horizontal"

    # -----------------------------------------------------------------------
    # Solver — tuned for A100-80GB PCIe (28 CPUs, 120 GB RAM)
    # -----------------------------------------------------------------------
    # A100-80GB comfortably fits 16 images/iter at 1024×1024 LSJ crops with AMP.
    # Effective batch = 16, matching the official Mask2Former LR (1e-4).
    # No gradient accumulation needed — removes accum overhead for ~15% speed gain.
    cfg.SOLVER.IMS_PER_BATCH           = 16

    _EFFECTIVE_ITERS = 100_000
    cfg.SOLVER.MAX_ITER                = _EFFECTIVE_ITERS

    # Official Mask2Former LR for bs16
    cfg.SOLVER.BASE_LR                 = 5e-5
    cfg.SOLVER.WEIGHT_DECAY            = 0.05

    cfg.SOLVER.OPTIMIZER             = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER   = 0.1   # lower LR for pretrained backbone

    # Cosine decay with linear warmup
    # 5000 warmup iters ≈ 2% of 267K-image run; avoids cold-start LR spikes.
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_FACTOR     = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS      = 1_000
    cfg.SOLVER.WARMUP_METHOD     = "linear"

    # Clip gradients to prevent spikes on fashion's long-tail distribution
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED       = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE     = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE    = 1.0     # official Mask2Former default; 0.01 was too aggressive
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE     = 2.0

    # -----------------------------------------------------------------------
    # Mixed precision (AMP / FP16)
    # -----------------------------------------------------------------------
    cfg.SOLVER.AMP.ENABLED = True

    # -----------------------------------------------------------------------
    # Checkpointing & evaluation
    # -----------------------------------------------------------------------
    # Checkpoint every 2500 iters (~53 min at 1.27s/iter).  Limits data loss
    # from NaN weight corruption to at most ~53 min of training.
    cfg.SOLVER.CHECKPOINT_PERIOD = 2_500
    cfg.TEST.EVAL_PERIOD          = 5_000

    # -----------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------
    # 28 CPUs / 120 GB RAM: 16 workers saturate the GPU without thrashing.
    # pin_memory: reduces CPU↔GPU transfer latency.
    # persistent_workers: avoids worker restart overhead between iterations.
    cfg.DATALOADER.NUM_WORKERS           = 16
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
    print(f"  CLIP_VALUE      : {cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE}")