#!/usr/bin/env python3
"""
run_training.py — Resilient training launcher with auto-restart.

Wraps training.py with:
  - Auto-restart on crash (OOM, NaN, DataLoader errors)
    - Restarts from latest checkpoint weights (optimizer state is intentionally reset)
  - Exponential backoff on repeated failures
  - GPU memory cleanup between retries
  - Logging of all crashes to a persistent log
  - Stops after MAX_ITER is reached or max retries exhausted

Usage:
    python run_training.py                          # uses defaults below
    nohup python run_training.py > train.log 2>&1 & # background, no tmux needed
"""

import datetime
import gc
import json
import os
import subprocess
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURE THESE — edit to match your server paths
#  RUN_VARIANT can be selected via env var RUN_VARIANT=run1|run2
#  BACKBONE can be selected via env var BACKBONE=SWIN_T|SWIN_B|SWIN_L|R50
# ═══════════════════════════════════════════════════════════════════════════
BASE_CONFIG = {
    "model_weights":      "./output_swin_boundary/model_final.pth",  # profile default may override
    "train_json":         "/ephemeral/training_data/annotations/instances_train.json",
    "val_json":           "/ephemeral/training_data/annotations/instances_val.json",
    "train_images":       "/ephemeral/training_data/images/train",
    "val_images":         "/ephemeral/training_data/images/val",
    "ims_per_batch":      12,
    "num_workers":        16,
    "grad_accum":         1,
    "max_iter":           5_000,
    "eval_period":        1_000,
    "checkpoint_period":  1_000,
    "base_lr":            "5e-6",
    # LR decay steps — mild annealing in the last 1500 iters
    "solver_steps":       "(3500,4500)",
    # Boundary precision correction knobs
    "boundary_weight":               "6.0",
    "mask_weight":                   "0.6",
    "boundary_fp_hard_weight":       "0.5",
    "boundary_fp_hard_ratio":        "0.01",
    "boundary_fp_warmup_iters":      "1000",
}

BACKBONE_PROFILES = {
    # Baseline precision-correction continuation from your current Swin-T run.
    "SWIN_T": {
        "backbone": "SWIN_T",
        "ims_per_batch": 12,
        "model_weights": "./output_swin_boundary/model_final.pth",
    },
    # Recommended upgrade path for 97-class fine-grained fashion.
    "SWIN_B": {
        "backbone": "SWIN_B",
        "ims_per_batch": 8,
        "model_weights": (
            "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
            "maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_f07440.pkl"
        ),
    },
    # Highest-capacity option; ensure weight path/url is reachable in your env.
    "SWIN_L": {
        "backbone": "SWIN_L",
        "ims_per_batch": 6,
        "model_weights": "swin_large_patch4_window12_384_22k.pkl",
    },
    "R50": {
        "backbone": "R50",
        "ims_per_batch": 12,
        "model_weights": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
    },
}

RUN_VARIANTS = {
    # Run 1: isolate FP correction only.
    "run1": {
        "output_dir": "./output_swin_boundary_precision_fp",
        "boundary_size_weight_enabled": "False",
    },
    # Run 2: re-enable size weighting only after Run 1 proves reduced bleeding.
    "run2": {
        "output_dir": "./output_swin_boundary_precision_fp_sizew",
        "boundary_size_weight_enabled": "True",
    },
}

RUN_VARIANT = os.environ.get("RUN_VARIANT", "run1").strip().lower()
if RUN_VARIANT not in RUN_VARIANTS:
    valid = ", ".join(sorted(RUN_VARIANTS.keys()))
    raise ValueError(f"Invalid RUN_VARIANT='{RUN_VARIANT}'. Expected one of: {valid}")

BACKBONE = os.environ.get("BACKBONE", "SWIN_T").strip().upper()
if BACKBONE not in BACKBONE_PROFILES:
    valid = ", ".join(sorted(BACKBONE_PROFILES.keys()))
    raise ValueError(f"Invalid BACKBONE='{BACKBONE}'. Expected one of: {valid}")

# Bias-correction phase should stay on Swin-T unless user explicitly forces upgrade.
FORCE_BACKBONE_UPGRADE = os.environ.get("FORCE_BACKBONE_UPGRADE", "0").strip() == "1"
if RUN_VARIANT in {"run1", "run2"} and BACKBONE != "SWIN_T" and not FORCE_BACKBONE_UPGRADE:
    raise ValueError(
        "Bias-correction runs are locked to BACKBONE=SWIN_T by default. "
        "Set FORCE_BACKBONE_UPGRADE=1 only after stability/precision targets are met."
    )

CONFIG = {**BASE_CONFIG, **BACKBONE_PROFILES[BACKBONE], **RUN_VARIANTS[RUN_VARIANT]}

# Retry settings
MAX_RETRIES      = 20        # total restart attempts before giving up
BASE_WAIT_SECS   = 30        # wait after first crash
MAX_WAIT_SECS    = 300       # cap backoff at 5 minutes
CRASH_LOG        = os.path.join(CONFIG["output_dir"], "crash_log.jsonl")
# ═══════════════════════════════════════════════════════════════════════════


def resolve_start_weights() -> str:
    """Return seed checkpoint for this attempt.

    Priority:
      1) latest checkpoint in output_dir (weights-only restart)
      2) configured seed checkpoint in CONFIG["model_weights"]
    """
    ckpt_file = os.path.join(CONFIG["output_dir"], "last_checkpoint")
    if os.path.isfile(ckpt_file):
        with open(ckpt_file) as f:
            ckpt_path = f.read().strip()
        if ckpt_path:
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(CONFIG["output_dir"], ckpt_path)
            if os.path.isfile(ckpt_path):
                return ckpt_path
    return CONFIG["model_weights"]


def build_command() -> list:
    """Build the training.py subprocess command."""
    model_weights = resolve_start_weights()
    cmd = [
        sys.executable, "training.py",
        # NO --resume: optimizer state intentionally reset because the loss fn changed.
        # Weights are loaded via MODEL.WEIGHTS below instead.
        "--train-only",
        "--backbone",        CONFIG["backbone"],
        "--output-dir",       CONFIG["output_dir"],
        "--train-json",       CONFIG["train_json"],
        "--val-json",         CONFIG["val_json"],
        "--train-images",     CONFIG["train_images"],
        "--val-images",       CONFIG["val_images"],
        "--ims-per-batch",    str(CONFIG["ims_per_batch"]),
        "--num-workers",      str(CONFIG["num_workers"]),
        "--grad-accum-steps", str(CONFIG["grad_accum"]),
        "--max-iter",         str(CONFIG["max_iter"]),
        # Detectron2 opts
        "MODEL.WEIGHTS",                    model_weights,
        "SOLVER.BASE_LR",                   CONFIG["base_lr"],
        "SOLVER.STEPS",                     CONFIG["solver_steps"],
        "TEST.EVAL_PERIOD",                 str(CONFIG["eval_period"]),
        "SOLVER.CHECKPOINT_PERIOD",         str(CONFIG["checkpoint_period"]),
        "MODEL.MASK_FORMER.BOUNDARY_WEIGHT", CONFIG["boundary_weight"],
        "MODEL.MASK_FORMER.MASK_WEIGHT",    CONFIG["mask_weight"],
        "MODEL.MASK_FORMER.BOUNDARY_FP_HARD_WEIGHT", CONFIG["boundary_fp_hard_weight"],
        "MODEL.MASK_FORMER.BOUNDARY_FP_HARD_RATIO", CONFIG["boundary_fp_hard_ratio"],
        "MODEL.MASK_FORMER.BOUNDARY_FP_WARMUP_ITERS", CONFIG["boundary_fp_warmup_iters"],
        "MODEL.MASK_FORMER.BOUNDARY_SIZE_WEIGHT_ENABLED", CONFIG["boundary_size_weight_enabled"],
    ]
    return cmd


def get_last_iter() -> int:
    """Read the last checkpoint iteration from output dir."""
    ckpt_file = os.path.join(CONFIG["output_dir"], "last_checkpoint")
    if not os.path.isfile(ckpt_file):
        return 0
    with open(ckpt_file) as f:
        ckpt_path = f.read().strip()
    # e.g. "model_0009999.pth" → 9999
    base = os.path.basename(ckpt_path)
    if "model_" in base:
        try:
            return int(base.split("model_")[1].split(".")[0])
        except (IndexError, ValueError):
            pass
    if "model_final" in base:
        return CONFIG["max_iter"]
    return 0


def log_crash(attempt: int, exit_code: int, elapsed: float, last_iter: int, stderr_tail: str):
    """Append crash info to a JSONL log file."""
    os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "attempt":   attempt,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 1),
        "last_iter": last_iter,
        "stderr_tail": stderr_tail[-2000:],  # last 2KB
    }
    with open(CRASH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cleanup_gpu():
    """Best-effort GPU memory cleanup between retries."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def preflight_check() -> None:
    """Validate all required paths exist before entering the retry loop.

    Unrecoverable config errors (missing files / directories) must abort
    immediately with a clear message instead of burning through MAX_RETRIES
    on fast crashes that will never self-heal.
    """
    required = {
        "train_json":    ("file", CONFIG["train_json"]),
        "val_json":      ("file", CONFIG["val_json"]),
        "train_images":  ("dir",  CONFIG["train_images"]),
        "val_images":    ("dir",  CONFIG["val_images"]),
    }
    errors = []
    for key, (kind, path) in required.items():
        if kind == "file" and not os.path.isfile(path):
            errors.append(f"  ✗ [{key}] file not found: {path}")
        elif kind == "dir" and not os.path.isdir(path):
            errors.append(f"  ✗ [{key}] directory not found: {path}")

    if errors:
        print("\n🛑 Pre-flight check failed — fix these before starting training:\n")
        for e in errors:
            print(e)
        print(
            "\nEdit CONFIG at the top of run_training.py to point to the correct paths.\n"
        )
        sys.exit(1)

    # CUDA availability — must fail here, not inside the retry loop.
    # If no GPU is visible the subprocess will crash after ~350s of model
    # loading, bypass the fast-crash counter, and retry indefinitely.
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n🛑 Pre-flight check failed — no CUDA GPU is visible.\n")
            print("  Diagnosis steps:")
            print("    1. nvidia-smi                        # verify driver sees the GPU")
            print("    2. echo $CUDA_VISIBLE_DEVICES         # must NOT be empty string or -1")
            print("    3. python3 -c 'import torch; print(torch.cuda.device_count())'")
            print("    4. If using a venv: deactivate && source .venv/bin/activate && nvidia-smi")
            print("    5. If in a container: check --gpus flag or device passthrough\n")
            sys.exit(1)
        gpu_count = torch.cuda.device_count()
        gpu_name  = torch.cuda.get_device_name(0)
        print(f"✅ Pre-flight check passed — {gpu_count} GPU(s) visible: {gpu_name}")
    except ImportError:
        print("⚠️  torch not importable from launcher — skipping CUDA check.")


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"\n[launcher] RUN_VARIANT={RUN_VARIANT}")
    print(f"[launcher] BACKBONE={BACKBONE}")
    print(f"[launcher] output_dir={CONFIG['output_dir']}")

    # Fail fast on missing files/dirs — don't waste retry budget on config errors.
    preflight_check()

    consecutive_fast_crashes = 0  # crashes < 60s likely mean a code bug, not transient

    for attempt in range(1, MAX_RETRIES + 1):
        last_iter = get_last_iter()

        if last_iter >= CONFIG["max_iter"]:
            print(f"\n✅ Training complete! Reached iter {last_iter}/{CONFIG['max_iter']}.")
            return

        wait = min(BASE_WAIT_SECS * (2 ** (attempt - 2)), MAX_WAIT_SECS) if attempt > 1 else 0

        print(f"\n{'='*60}")
        print(f"  Attempt {attempt}/{MAX_RETRIES}")
        print(f"  Restarting from iter {last_iter}/{CONFIG['max_iter']} (weights-only)")
        print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if wait > 0:
            print(f"  Waiting {wait}s before restart...")
            time.sleep(wait)
        print(f"{'='*60}\n")

        cmd = build_command()
        print(f"  CMD: {' '.join(cmd[:6])} ...\n")

        start = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,     # stream stdout live
            stderr=subprocess.PIPE, # capture stderr for crash log
            text=True,
            bufsize=1,
        )

        # Stream stderr live while also capturing it
        stderr_lines = []
        try:
            for line in proc.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)
                # Keep only last 200 lines to limit memory
                if len(stderr_lines) > 200:
                    stderr_lines.pop(0)
        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C detected — stopping training.")
            proc.terminate()
            proc.wait(timeout=30)
            return

        exit_code = proc.wait()
        elapsed = time.time() - start
        stderr_tail = "".join(stderr_lines)

        # Success
        if exit_code == 0:
            final_iter = get_last_iter()
            print(f"\n✅ Training finished successfully at iter {final_iter}.")
            return

        # Crash
        log_crash(attempt, exit_code, elapsed, last_iter, stderr_tail)
        cleanup_gpu()

        new_iter = get_last_iter()
        progress = new_iter - last_iter

        print(f"\n❌ Crash! exit_code={exit_code}, ran for {elapsed:.0f}s, "
              f"progress: {last_iter}→{new_iter} (+{progress} iters)")

        # Detect fast crashes (likely code bugs, not transient)
        if elapsed < 60:
            consecutive_fast_crashes += 1
            if consecutive_fast_crashes >= 3:
                print(f"\n🛑 3 consecutive fast crashes (<60s each). Likely a code bug, not transient.")
                print(f"   Check: {CRASH_LOG}")
                return
        else:
            consecutive_fast_crashes = 0

        # Detect stuck (no progress across retries)
        if attempt >= 3 and progress == 0:
            print(f"\n🛑 No progress in last attempt. Check crash log: {CRASH_LOG}")
            # Continue anyway — the next restart may fix it (different random batch)

    print(f"\n🛑 Exhausted all {MAX_RETRIES} retries. Last iter: {get_last_iter()}")
    print(f"   Crash log: {CRASH_LOG}")


if __name__ == "__main__":
    main()
