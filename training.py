"""Mask2Former training entrypoint with strict COCO preflight checks."""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

# ── NumPy/Torch compatibility guard ──────────────────────────────────────────
# PyTorch 2.2.x wheels were compiled against NumPy 1.x ABI.
# If NumPy 2.x is installed, torch.from_numpy() crashes at dataset loading.
try:
    import numpy as np
    _np_major = int(np.__version__.split(".")[0])
    if _np_major >= 2:
        print(
            f"\n*** FATAL: numpy {np.__version__} detected but PyTorch requires numpy<2.\n"
            f"*** Fix:   pip install 'numpy>=1.26,<2.0'\n",
            file=sys.stderr,
        )
        sys.exit(1)
except ImportError:
    print("*** FATAL: numpy is not installed. pip install 'numpy>=1.26,<2.0'", file=sys.stderr)
    sys.exit(1)

import torch

# ── Detectron2 / numpy compat monkey-patch ──────────────────────────────────
# np.bool was removed in numpy 1.24; detectron2 v0.6 still uses it in masks.py
import numpy as _np
_np.bool = bool  # safe: bool is what np.bool always was

# ── A100 / Ampere+ optimizations ─────────────────────────────────────────
# TF32 gives ~2× throughput on matmuls with negligible precision loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Auto-tune convolution algorithms for best perf on this GPU
torch.backends.cudnn.benchmark = True

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import AMPTrainer, DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

from config_setup import build_cfg
from register_dataset import get_thing_classes, register_fashion_datasets, run_preflight_checks

logger = logging.getLogger("mask2former.train")


class FashionTrainer(DefaultTrainer):
    """Trainer helpers for evaluator, data loader, and optimizer setup."""

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder: Optional[str] = None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        return DatasetEvaluators(
            [
                COCOEvaluator(
                    dataset_name,
                    tasks={"segm"},
                    distributed=True,
                    output_dir=output_folder,
                    use_fast_impl=True,
                )
            ]
        )

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        from detectron2.data import DatasetMapper

        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice",
            ),
            T.RandomFlip(horizontal=True, vertical=False),
        ]

        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=augmentations,
            use_instance_mask=True,
            instance_mask_format="bitmask",
        )
        return build_detection_train_loader(
            cfg,
            mapper=mapper,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module):
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer):
        return build_lr_scheduler(cfg, optimizer)


class GradAccumAMPTrainer(AMPTrainer):
    """AMP trainer with configurable gradient accumulation."""

    def __init__(self, model, data_loader, optimizer, accum_steps: int = 1):
        super().__init__(model, data_loader, optimizer)
        self.accum_steps = max(1, int(accum_steps))
        self.clip_cfg = None

    def run_step(self):
        assert self.model.training, "Model was changed to eval mode during training."
        start = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        loss_dict_accum: Dict[str, float] = {}

        for _ in range(self.accum_steps):
            data = next(self._data_loader_iter)
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    loss_dict = self.model(data)
                    losses = sum(loss_dict.values()) / self.accum_steps
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Skipping bad batch (exception): {e}")
                self.optimizer.zero_grad(set_to_none=True)
                data_time = time.perf_counter() - start
                self.storage.put_scalars(data_time=data_time)
                return

            # NaN/Inf guard: skip batch before backward() corrupts weights
            if not torch.isfinite(losses):
                logger.warning(
                    f"[iter {self.storage.iter}] Non-finite loss={losses.item():.4f}, "
                    f"skipping batch. loss_dict={  {k: v.item() for k, v in loss_dict.items()} }"
                )
                self.optimizer.zero_grad(set_to_none=True)
                data_time = time.perf_counter() - start
                self.storage.put_scalars(data_time=data_time)
                return

            self.grad_scaler.scale(losses).backward()
            for key, val in loss_dict.items():
                loss_dict_accum[key] = loss_dict_accum.get(key, 0.0) + (val.detach().item() / self.accum_steps)

        if self.clip_cfg is not None and self.clip_cfg.ENABLED:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_cfg.CLIP_VALUE)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        data_time = time.perf_counter() - start
        total_loss = sum(loss_dict_accum.values())
        self.storage.put_scalars(total_loss=total_loss, data_time=data_time, **loss_dict_accum)


class GPUMemoryHook(hooks.HookBase):
    """Periodically logs GPU memory stats to help diagnose OOM regressions."""

    def __init__(self, period: int = 100):
        self._period = max(1, int(period))

    def after_step(self):
        if not torch.cuda.is_available():
            return
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return

        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        self.trainer.storage.put_scalars(
            gpu_mem_allocated_mb=allocated_mb,
            gpu_mem_reserved_mb=reserved_mb,
            gpu_mem_max_allocated_mb=max_allocated_mb,
        )


def evaluate(cfg: CfgNode, model) -> Dict:
    """Evaluate on all configured test datasets."""
    results: Dict = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = FashionTrainer.build_evaluator(cfg, dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation %s: %s", dataset_name, results_i)
    return results


def train(
    cfg: CfgNode,
    resume: bool,
    grad_accum_steps: int,
    freeze_backbone: bool = False,
    log_gpu_mem_interval: int = 100,
) -> None:
    """Run full training with checkpointing, evaluation, and logging."""
    logger.info("Starting training with grad_accum_steps=%d", grad_accum_steps)

    model = build_model(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))

    if freeze_backbone:
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen: training head/decoder only.")
        else:
            logger.warning("--freeze-backbone was set, but model has no 'backbone' attribute.")

    optimizer = FashionTrainer.build_optimizer(cfg, model)
    scheduler = FashionTrainer.build_lr_scheduler(cfg, optimizer)
    data_loader = FashionTrainer.build_train_loader(cfg)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    if resume:
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1
    else:
        checkpointer.load(cfg.MODEL.WEIGHTS)
        start_iter = 0

    trainer = GradAccumAMPTrainer(model, data_loader, optimizer, accum_steps=grad_accum_steps)
    trainer.clip_cfg = cfg.SOLVER.CLIP_GRADIENTS

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=scheduler),
            hooks.PeriodicCheckpointer(checkpointer, period=cfg.SOLVER.CHECKPOINT_PERIOD),
            hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: evaluate(cfg, model)),
            GPUMemoryHook(period=log_gpu_mem_interval),
            hooks.PeriodicWriter(
                [
                    CommonMetricPrinter(cfg.SOLVER.MAX_ITER),
                    JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                    TensorboardXWriter(cfg.OUTPUT_DIR),
                ],
                period=20,
            ),
        ]
    )

    with EventStorage(start_iter) as storage:
        trainer.storage = storage
        trainer.train(start_iter, cfg.SOLVER.MAX_ITER)

    if comm.is_main_process():
        final_results = evaluate(cfg, model)
        for dataset_name, dataset_results in final_results.items():
            logger.info("Final evaluation summary for dataset=%s", dataset_name)
            if isinstance(dataset_results, dict):
                print_csv_format(dataset_results)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask2Former fashion training")
    parser.add_argument("--output-dir", default="./output", help="Output directory for logs/checkpoints")
    parser.add_argument("--backbone", default="R50", choices=["R50", "SWIN_T"], help="Backbone model")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("--max-iter", type=int, default=None, help="Override max iterations")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (1 = no accumulation)")
    parser.add_argument("--ims-per-batch", type=int, default=None, help="Override SOLVER.IMS_PER_BATCH")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DATALOADER.NUM_WORKERS")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short 2000-iteration sanity test")
    parser.add_argument("--preflight-only", action="store_true", help="Only run dataset validation and exit")
    parser.add_argument("--data-root", default=None, help="Dataset root with images/ and annotations/")
    parser.add_argument("--train-json", default=None, help="Path to train COCO JSON")
    parser.add_argument("--val-json", default=None, help="Path to val COCO JSON")
    parser.add_argument("--train-images", default=None, help="Path to train image directory")
    parser.add_argument("--val-images", default=None, help="Path to val image directory")
    parser.add_argument("--classes-file", default=None, help="Path to class list file (1 class per line)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone for fast warmup")
    parser.add_argument(
        "--log-gpu-mem-interval",
        type=int,
        default=100,
        help="Log GPU memory stats every N iterations",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional Detectron2 overrides")
    return parser


def main(args: argparse.Namespace) -> None:
    # Allow per-run dataset/class overrides without editing source files.
    if args.data_root:
        os.environ["FASHION_DATA_ROOT"] = args.data_root
    if args.train_json:
        os.environ["FASHION_TRAIN_JSON"] = args.train_json
    if args.val_json:
        os.environ["FASHION_VAL_JSON"] = args.val_json
    if args.train_images:
        os.environ["FASHION_TRAIN_IMAGES"] = args.train_images
    if args.val_images:
        os.environ["FASHION_VAL_IMAGES"] = args.val_images
    if args.classes_file:
        os.environ["FASHION_CLASSES_FILE"] = args.classes_file

    run_preflight_checks()
    if args.preflight_only:
        print("[training] Preflight checks passed.")
        return

    register_fashion_datasets()
    thing_classes = get_thing_classes()
    print(f"[training] classes={len(thing_classes)}")
    print(f"[training] first_classes={thing_classes[:10]}")

    cfg = build_cfg(
        output_dir=args.output_dir,
        resume=args.resume,
        backbone=args.backbone,
        num_classes=len(thing_classes),
    )
    cfg.defrost()

    if args.smoke_test:
        cfg.SOLVER.MAX_ITER = 2000
        cfg.TEST.EVAL_PERIOD = min(cfg.TEST.EVAL_PERIOD, 1000)
        cfg.SOLVER.CHECKPOINT_PERIOD = min(cfg.SOLVER.CHECKPOINT_PERIOD, 1000)
    if args.max_iter is not None:
        cfg.SOLVER.MAX_ITER = int(args.max_iter)
    if args.ims_per_batch is not None:
        cfg.SOLVER.IMS_PER_BATCH = int(args.ims_per_batch)
    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    train(
        cfg,
        resume=args.resume,
        grad_accum_steps=args.grad_accum_steps,
        freeze_backbone=args.freeze_backbone,
        log_gpu_mem_interval=args.log_gpu_mem_interval,
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
