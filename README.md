# Mask2Former — Fashion Instance Segmentation

Production-grade **Mask2Former** training pipeline for high-fidelity fashion apparel segmentation across **97 garment categories**. Built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [Mask2Former](https://github.com/facebookresearch/Mask2Former) with custom stability hardening, edge-aware validation metrics, and crash-resilient training orchestration.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  run_training.py (Orchestrator)             │
│  Auto-restart · GPU cleanup · Logging       │
├─────────────────────────────────────────────┤
│  training.py (Training Loop)                │
│  BFloat16 AMP · Gradient checks · Hooks     │
├─────────────────────────────────────────────┤
│  config_setup.py     │  register_dataset.py │
│  Solver / Model cfg  │  COCO registration   │
├──────────────────────┴──────────────────────┤
│  validation_utils.py                        │
│  Boundary IoU · BFScore · FP tracking       │
│  Stratified subset · Error overlays         │
├─────────────────────────────────────────────┤
│  inference.py                               │
│  Single-image / batch inference             │
└─────────────────────────────────────────────┘
```

## Key Features

- **Swin-Tiny Backbone** — Global attention for capturing thin fashion structures (straps, chains, collars)
- **BFloat16 Training** — Eliminates FP16 overflow (`65,504` limit) on A100/H100 GPUs
- **NaN/Inf Defense** — Post-unscale gradient validation skips degenerate batches automatically
- **Crash-Resilient Wrapper** — `run_training.py` handles OOM, NaN crashes with auto-restart (up to 20 attempts)
- **Edge-Aware Validation** — Custom `ValidationHook` with Boundary IoU (strict + loose), BFScore, and small-object tracking
- **Stratified Evaluation** — 200-image fixed subset balanced across small objects, occlusions, and normal scenes
- **Visual Debugging** — Tricolor error overlays pushed to TensorBoard: Green=GT, Red=Pred, Blue=Error

---

## Quick Start

### 1. Environment Setup

**Requirements:** Python 3.10 · CUDA 12.2 · Ubuntu 22.04+

```bash
# Clone and setup
git clone <repo-url> mask2fromer && cd mask2fromer
bash setup.sh
source .venv/bin/activate
```

The `setup.sh` script installs PyTorch, Detectron2, Mask2Former, builds custom CUDA ops (MSDeformAttn), and patches numpy/Pillow compatibility issues automatically.

### 2. Dataset Preparation

Organize your COCO-format fashion dataset:

```
/path/to/training_data/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train_images/
│   └── *.jpg
└── val_images/
    └── *.jpg
```

Set environment variables (or pass as CLI args):

```bash
export FASHION_DATA_ROOT=/path/to/training_data
export FASHION_TRAIN_JSON=/path/to/training_data/annotations/instances_train.json
export FASHION_VAL_JSON=/path/to/training_data/annotations/instances_val.json
export FASHION_TRAIN_IMAGES=/path/to/training_data/train_images
export FASHION_VAL_IMAGES=/path/to/training_data/val_images
```

### 3. Training

```bash
# Full training (recommended — handles crashes automatically)
python run_training.py

# Resume from checkpoint
python run_training.py  # auto-detects last_checkpoint

# Direct training (no wrapper)
python training.py --backbone SWIN_T --output-dir ./output_swin_fashion --resume
```

### 4. Monitor Training

```bash
tensorboard --logdir ./output_swin_fashion --port 6006
```

**Tracked metrics:**

| Metric | TensorBoard Key | Description |
|---|---|---|
| Foreground IoU | `val/fg_iou` | Overall mask coverage accuracy |
| Foreground Dice | `val/fg_dice` | Overlap coefficient |
| Boundary IoU (Strict) | `val/bound_iou_strict` | 1px hard-cut edge precision |
| Boundary IoU (Loose) | `val/bound_iou_loose` | Dynamic-kernel edge tracking |
| BFScore | `val/bfscore` | Boundary F-score (3px tolerance) |
| Instance IoU | `val/inst_iou` | Per-instance match quality |
| False Positive Rate | `val/fp_rate` | Area-weighted mask hallucination |
| Small Object BFScore | `val/small_obj_bfscore` | Edge quality for items < 5% area |

### 5. Inference

```bash
python inference.py --input ./test_image.jpg --output ./results/ --config ./output_swin_fashion/config.yaml --weights ./output_swin_fashion/model_final.pth
```

---

## Project Structure

```
mask2fromer/
├── training.py                 # Core Detectron2 training loop with AMP + gradient defense
├── run_training.py             # Crash-resilient orchestrator with auto-restart
├── config_setup.py             # Solver, model, and augmentation configuration
├── register_dataset.py         # COCO dataset registration + preflight validation
├── validation_utils.py         # Custom ValidationHook (Boundary IoU, BFScore, overlays)
├── inference.py                # Single/batch inference pipeline
├── smoke_test_validation.py    # Offline validation hook tester (no GPU training cost)
├── setup.sh                    # One-shot environment installer
├── requirements.txt            # Python dependencies
└── Mask2Former/                # Facebook Mask2Former repo (cloned by setup.sh)
```

---

## Configuration

All training hyperparameters are centralized in `config_setup.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `BASE_LR` | `5e-5` | Safe for Swin backbone fine-tuning |
| `LR_SCHEDULE` | WarmupMultiStep (drops at 20k, 40k) | Standard Mask2Former schedule |
| `CLIP_GRADIENTS` | `1.0` | Prevents gradient explosion |
| `AMP` | BFloat16 | No overflow risk on modern GPUs |
| `MASK_WEIGHT` | `6.0` | Edge-focused loss balancing |
| `DICE_WEIGHT` | `4.0` | Complementary overlap loss |
| `NUM_OBJECT_QUERIES` | `200` | Headroom for dense multi-garment scenes |
| `DEC_LAYERS` | `12` | Full decoder depth |
| `IMAGE_SIZE` | `1024` | LSJ augmentation target |
| `MIN_SCALE / MAX_SCALE` | `0.5 / 1.5` | Controlled augmentation bounds |

---

## Validation System

The custom `ValidationHook` replaces standard COCO evaluation with a fast, edge-focused monitoring system:

- **Stratified 200-image subset** — 50 small-object + 50 occlusion + 100 normal images, seeded deterministically
- **50 images per eval cycle** — Rolling window covers full subset every 4 cycles
- **Adaptive frequency** — 2000 iters (0–20k) → 1000 iters (20k–40k) → 500 iters (40k+)
- **JSON cache** — Subset saved to disk after first build, eliminating 11GB RAM spike on restarts
- **GPU-native metrics** — All boundary computations use `F.max_pool2d` for dilation/erosion

### Smoke Test

Validate the hook works without consuming training iterations:

```bash
python smoke_test_validation.py
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `NaN loss` at early iters | FP16 overflow | Already using BFloat16 — if persists, reduce `BASE_LR` |
| `IndexError: tuple index out of range` | `cfg.DATASETS.TEST` wiped by `--train-only` | Fixed — TEST tuple preserved for ValidationHook |
| `TypeError: unhashable type: 'dict'` | Raw dataset list passed to loader | Fixed — subset registered via `DatasetCatalog` |
| `AssertionError: Attribute 'name'` | Metadata copy includes reserved `name` field | Fixed — `name` stripped before copy |
| All validation metrics = 0 | Score threshold too high mid-training | Threshold set to `0.1` for monitoring (use `0.5+` for final eval) |
| `RuntimeError: device mismatch` | Detectron2 moves outputs to CPU post-inference | Fixed — selective `.to("cuda")` on required tensors only |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX A4000 (16GB) | A100 (40GB+) |
| RAM | 32GB | 64GB+ |
| Storage | 100GB | 250GB+ (for checkpoints) |
| CUDA | 12.1 | 12.2+ |

---

## License

**Proprietary** — © 2026 MnemonicX Tech. All Rights Reserved.

This software is confidential and proprietary. Unauthorized copying, distribution, or use is strictly prohibited. See [LICENSE](./LICENSE) for full terms.

### Third-Party Acknowledgements

This project depends on the following open-source components:

| Component | License |
|---|---|
| [Detectron2](https://github.com/facebookresearch/detectron2) | Apache 2.0 |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3 |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | MIT |
