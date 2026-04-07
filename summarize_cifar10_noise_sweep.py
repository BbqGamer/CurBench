#!/usr/bin/env python3
"""Summarize CIFAR-10 noise sweep results.

Reads the local run logs produced by run_cifar10_noise_sweep.py and computes
mean/std test accuracy across seeds for each method and noise level.

Optionally logs the summary figure + table to W&B.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SEEDS = [42, 666, 777, 888, 999]
NOISE_PS = [round(i * 0.1, 1) for i in range(9)]
METHODS = ["base", "spl", "ttcl"]


ACC_RE = re.compile(r"Final Test Acc = ([0-9]*\.?[0-9]+)")


def build_data_name(noise_p: float) -> str:
    return "cifar10" if noise_p == 0.0 else f"cifar10-noise-{noise_p:.1f}"


def run_dir(method: str, data_name: str, net: str, epochs: int, seed: int) -> Path:
    return Path("runs") / f"{method}-{data_name}-{net}-{epochs}-{seed}"


def parse_test_acc(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = ACC_RE.search(line)
        if m:
            last = float(m.group(1))
    return last


def maybe_log_wandb(summary_rows, fig_path: Path, project: str, entity: str | None, mode: str, group: str):
    try:
        import wandb
    except Exception:
        print("wandb is unavailable; skipping W&B logging")
        return

    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        group=group,
        job_type="summary",
        name="cifar10-noise-sweep-summary",
        config={"dataset": "cifar10", "net": "resnet18"},
    )
    try:
        table = wandb.Table(columns=["method", "noise_p", "mean_acc", "std_acc", "n"])
        for row in summary_rows:
            table.add_data(row["method"], row["noise_p"], row["mean_acc"], row["std_acc"], row["n"])
        run.log({
            "summary/table": table,
            "summary/plot": wandb.Image(str(fig_path)),
        })
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--net", type=str, default="resnet18")
    parser.add_argument("--out_csv", type=str, default="runs/cifar10-noise-sweep-summary.csv")
    parser.add_argument("--out_fig", type=str, default="runs/cifar10-noise-sweep-summary.png")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="CurBench")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_group", type=str, default="cifar10-noise-sweep-resnet18")
    args = parser.parse_args()

    all_rows = []
    for method in METHODS:
        for noise_p in NOISE_PS:
            data_name = build_data_name(noise_p)
            accs = []
            for seed in SEEDS:
                log_path = run_dir(method, data_name, args.net, args.epochs, seed) / "train.log"
                acc = parse_test_acc(log_path)
                if acc is None:
                    print(f"Missing: {log_path}")
                    continue
                accs.append(acc)
            if accs:
                all_rows.append({
                    "method": method,
                    "noise_p": noise_p,
                    "mean_acc": mean(accs),
                    "std_acc": pstdev(accs) if len(accs) > 1 else 0.0,
                    "n": len(accs),
                })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "noise_p", "mean_acc", "std_acc", "n"])
        writer.writeheader()
        writer.writerows(all_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable; wrote CSV only: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        rows = sorted((r for r in all_rows if r["method"] == method), key=lambda r: r["noise_p"])
        if not rows:
            continue
        xs = [r["noise_p"] for r in rows]
        ys = [r["mean_acc"] for r in rows]
        ss = [r["std_acc"] for r in rows]
        ax.plot(xs, ys, marker="o", label=method)
        ax.fill_between(xs, [y - s for y, s in zip(ys, ss)], [y + s for y, s in zip(ys, ss)], alpha=0.2)

    ax.set_xlabel("noise-p")
    ax.set_ylabel("test accuracy")
    ax.set_title("CIFAR-10 noise sweep: mean ± std over seeds")
    ax.set_xticks(NOISE_PS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_fig}")

    if args.wandb:
        maybe_log_wandb(all_rows, out_fig, args.wandb_project, args.wandb_entity, args.wandb_mode, args.wandb_group)


if __name__ == "__main__":
    main()
