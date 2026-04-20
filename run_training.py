#!/usr/bin/env python3
"""
run_training.py — Resilient training launcher with auto-restart.

Wraps training.py with:
  - Auto-restart on crash (OOM, NaN, DataLoader errors)
  - Always resumes from last checkpoint
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
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    "output_dir":    "./output_97cls_100k",
    "classes_file":  "/ephemeral/training_data/classes.txt",
    "train_json":    "/ephemeral/training_data/annotations/instances_train.json",
    "val_json":      "/ephemeral/training_data/annotations/instances_val.json",
    "train_images":  "/ephemeral/training_data/images/train",
    "val_images":    "/ephemeral/training_data/images/val",
    "ims_per_batch": 8,
    "num_workers":   16,
    "grad_accum":    2,
    "max_iter":      100_000,
    "eval_period":   999_999,       # skip eval during training (run offline later)
    "checkpoint_period": 10_000,
    "base_lr":       "1e-4",
}

# Retry settings
MAX_RETRIES      = 20        # total restart attempts before giving up
BASE_WAIT_SECS   = 30        # wait after first crash
MAX_WAIT_SECS    = 300       # cap backoff at 5 minutes
CRASH_LOG        = os.path.join(CONFIG["output_dir"], "crash_log.jsonl")
# ═══════════════════════════════════════════════════════════════════════════


def build_command() -> list:
    """Build the training.py subprocess command."""
    cmd = [
        sys.executable, "training.py",
        "--resume",
        "--output-dir",      CONFIG["output_dir"],
        "--classes-file",    CONFIG["classes_file"],
        "--train-json",      CONFIG["train_json"],
        "--val-json",        CONFIG["val_json"],
        "--train-images",    CONFIG["train_images"],
        "--val-images",      CONFIG["val_images"],
        "--ims-per-batch",   str(CONFIG["ims_per_batch"]),
        "--num-workers",     str(CONFIG["num_workers"]),
        "--grad-accum-steps", str(CONFIG["grad_accum"]),
        "--max-iter",        str(CONFIG["max_iter"]),
        # Detectron2 overrides via opts
        f"SOLVER.BASE_LR",         CONFIG["base_lr"],
        f"TEST.EVAL_PERIOD",       str(CONFIG["eval_period"]),
        f"SOLVER.CHECKPOINT_PERIOD", str(CONFIG["checkpoint_period"]),
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


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Set env for reduced fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    consecutive_fast_crashes = 0  # crashes < 60s likely mean a code bug, not transient

    for attempt in range(1, MAX_RETRIES + 1):
        last_iter = get_last_iter()

        if last_iter >= CONFIG["max_iter"]:
            print(f"\n✅ Training complete! Reached iter {last_iter}/{CONFIG['max_iter']}.")
            return

        wait = min(BASE_WAIT_SECS * (2 ** (attempt - 2)), MAX_WAIT_SECS) if attempt > 1 else 0

        print(f"\n{'='*60}")
        print(f"  Attempt {attempt}/{MAX_RETRIES}")
        print(f"  Resuming from iter {last_iter}/{CONFIG['max_iter']}")
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
