#!/usr/bin/env python3
"""Summarize Detectron2 training metrics for quick run-health checks."""

import argparse
import json
import math
import os
from statistics import median
from typing import Dict, List, Tuple


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _load_metrics(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping invalid JSON line {ln}")
                continue
            if isinstance(d, dict):
                rows.append(d)
    return rows


def _split_rows(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train_rows = [r for r in rows if "total_loss" in r]

    eval_rows: List[Dict] = []
    for r in rows:
        if any("/AP" in k for k in r.keys()):
            eval_rows.append(r)
    return train_rows, eval_rows


def _best_ap(eval_rows: List[Dict]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for r in eval_rows:
        for k, v in r.items():
            if "/AP" not in k:
                continue
            if not _is_finite_number(v):
                continue
            if k not in best or float(v) > best[k]:
                best[k] = float(v)
    return best


def _loss_trend(train_rows: List[Dict], window: int) -> Tuple[float, float, float]:
    vals = [float(r["total_loss"]) for r in train_rows if _is_finite_number(r.get("total_loss"))]
    if len(vals) < 2:
        return math.nan, math.nan, math.nan

    n = min(window, len(vals))
    head = vals[:n]
    tail = vals[-n:]
    head_med = median(head)
    tail_med = median(tail)

    pct = ((head_med - tail_med) / head_med * 100.0) if head_med > 0 else math.nan
    return head_med, tail_med, pct


def _count_bad_losses(train_rows: List[Dict]) -> int:
    bad = 0
    for r in train_rows:
        v = r.get("total_loss")
        if not _is_finite_number(v):
            bad += 1
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Detectron2 metrics.json")
    parser.add_argument("--metrics", default="output_sample5/metrics.json", help="Path to metrics.json")
    parser.add_argument("--window", type=int, default=50, help="Window size for loss trend median")
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"metrics file not found: {args.metrics}")

    rows = _load_metrics(args.metrics)
    train_rows, eval_rows = _split_rows(rows)

    print("=== Training Summary ===")
    print(f"metrics_file: {args.metrics}")
    print(f"rows_total: {len(rows)}")
    print(f"rows_train: {len(train_rows)}")
    print(f"rows_eval:  {len(eval_rows)}")

    if train_rows:
        last_train = train_rows[-1]
        print("\n--- Last Train Row ---")
        print(f"iteration:  {last_train.get('iteration')}")
        print(f"total_loss: {last_train.get('total_loss')}")
        print(f"lr:         {last_train.get('lr')}")
        print(f"data_time:  {last_train.get('data_time')}")

        head_med, tail_med, pct = _loss_trend(train_rows, args.window)
        print("\n--- Loss Trend ---")
        if math.isfinite(pct):
            print(f"head_median_loss: {head_med:.4f}")
            print(f"tail_median_loss: {tail_med:.4f}")
            print(f"improvement_pct:  {pct:.2f}%")
        else:
            print("not enough valid points for trend")

        bad_losses = _count_bad_losses(train_rows)
        print(f"bad_loss_rows:    {bad_losses}")
    else:
        print("\n[warn] no training rows found")

    best_ap = _best_ap(eval_rows)
    if best_ap:
        print("\n--- Best AP Metrics ---")
        for k in sorted(best_ap):
            print(f"{k}: {best_ap[k]:.4f}")
    else:
        print("\n[warn] no AP metrics found yet (did eval run?)")

    print("\n=== Verdict ===")
    healthy = True

    if not train_rows:
        healthy = False
        print("WARN: No train metrics found")

    if train_rows and _count_bad_losses(train_rows) > 0:
        healthy = False
        print("WARN: Non-finite losses detected")

    if train_rows:
        _, _, pct = _loss_trend(train_rows, args.window)
        if math.isfinite(pct) and pct <= 0:
            print("WARN: Loss not improving in selected window")

    if not eval_rows:
        print("INFO: No eval rows yet; may still be before first eval interval")

    if healthy:
        print("OK: Training metrics look structurally healthy")


if __name__ == "__main__":
    main()
