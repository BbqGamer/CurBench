#!/usr/bin/env python3
"""Run CIFAR-10 noise sweeps for base, SPL, and TTCL in a single W&B group.

Outer loop: seed
Inner loops: method -> noise level

TTCL uses the clean base run from seed 42 as its teacher.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 666, 777, 888, 999]
NOISE_PS = [round(i * 0.1, 1) for i in range(9)]
METHODS = ["base", "spl", "ttcl"]


def build_data_name(noise_p: float) -> str:
    if noise_p == 0.0:
        return "cifar10"
    return f"cifar10-noise-{noise_p:.1f}"


def build_run_dir(
    method: str, data_name: str, net: str, epochs: int, seed: int
) -> Path:
    return Path("runs") / f"{method}-{data_name}-{net}-{epochs}-{seed}"


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--net", type=str, default="resnet18")
    parser.add_argument("--project", type=str, default="CurBench")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--mode", type=str, default="online")
    parser.add_argument(
        "--group",
        type=str,
        default="cifar10-noise-sweep-resnet18",
        help="W&B group to put all runs into.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["CURBENCH_WANDB"] = "1"
    env["CURBENCH_WANDB_PROJECT"] = args.project
    env["CURBENCH_WANDB_MODE"] = args.mode
    env["CURBENCH_WANDB_GROUP"] = args.group
    if args.entity is not None:
        env["CURBENCH_WANDB_ENTITY"] = args.entity

    teacher_dir = build_run_dir("base", "cifar10", args.net, args.epochs, 42)
    if not teacher_dir.exists():
        teacher_cmd = [
            sys.executable,
            "examples/base.py",
            "--data",
            "cifar10",
            "--net",
            args.net,
            "--seed",
            "42",
            "--epochs",
            str(args.epochs),
            "--gpu",
            str(args.gpu),
            "--wandb",
            "--wandb_project",
            args.project,
            "--wandb_mode",
            args.mode,
            "--wandb_group",
            args.group,
            "--wandb_name",
            "seed42-base-p0.0-teacher",
        ]
        if args.entity is not None:
            teacher_cmd.extend(["--wandb_entity", args.entity])
        run_cmd(teacher_cmd, env=env, dry_run=args.dry_run)

    for seed in SEEDS:
        for noise_p in NOISE_PS:
            for method in METHODS:
                data_name = build_data_name(noise_p)
                run_dir = build_run_dir(method, data_name, args.net, args.epochs, seed)
                if run_dir.exists():
                    print(f"Already run: {run_dir}")
                    continue

                cmd = [
                    sys.executable,
                    f"examples/{method}.py",
                    "--data",
                    data_name,
                    "--net",
                    args.net,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--gpu",
                    str(args.gpu),
                    "--wandb",
                    "--wandb_project",
                    args.project,
                    "--wandb_mode",
                    args.mode,
                    "--wandb_group",
                    args.group,
                    "--wandb_name",
                    f"seed{seed}-{method}-p{noise_p:.1f}",
                ]
                if args.entity is not None:
                    cmd.extend(["--wandb_entity", args.entity])
                if method == "ttcl":
                    cmd.extend(["--teacher_dir", str(teacher_dir)])

                run_cmd(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
