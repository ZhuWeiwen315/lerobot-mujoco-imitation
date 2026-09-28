"""Reproducible ACT training with validation and periodic checkpoints."""

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import (
    make_act_pre_post_processors,
)

from .act_config import ACTExperimentConfig
from .checkpoint import save_act_checkpoint
from .evaluation import evaluate_act_policy


METRIC_FIELDS = [
    "step",
    "elapsed_seconds",
    "train_total_loss",
    "train_l1_loss",
    "train_kld_loss",
    "gradient_norm",
    "val_l1_loss",
]


def parse_args() -> argparse.Namespace:
    """Parse formal ACT experiment arguments."""
    parser = argparse.ArgumentParser(
        description="Train and validate ACT on Panda Lift datasets."
    )
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--val-dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New directory for metrics, config, and checkpoints.",
    )
    parser.add_argument(
        "--train-repo-id",
        default="zhuweiwen/panda-lift-train-v1",
    )
    parser.add_argument(
        "--val-repo-id",
        default="zhuweiwen/panda-lift-val-v1",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=0,
        help="Validation batches per evaluation; 0 means all.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or destructive experiment configurations."""
    positive_values = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
        "log_every": args.log_every,
        "val_every": args.val_every,
        "checkpoint_every": args.checkpoint_every,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.val_batches < 0:
        raise ValueError("--val-batches must be non-negative.")
    if not args.train_dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing training dataset: {args.train_dataset_root}"
        )
    if not args.val_dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing validation dataset: {args.val_dataset_root}"
        )
    if args.run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {args.run_dir}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and Torch CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_batch(
    loader: DataLoader,
    iterator: Any,
) -> tuple[dict[str, Any], Any]:
    """Read a batch, restarting the loader at an epoch boundary."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def make_loader(
    dataset: LeRobotDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Construct a DataLoader with safe worker settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def make_reference_chunk(
    policy: ACTPolicy,
    postprocessor,
    reference_batch: dict[str, Any],
) -> np.ndarray:
    """Generate a deterministic environment-scale action reference."""
    was_training = policy.training
    policy.eval()
    policy.reset()

    try:
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(
                reference_batch
            )
            environment_chunk = postprocessor(normalized_chunk)
    finally:
        policy.train(was_training)

    return environment_chunk.detach().cpu().numpy()


def checkpoint_path(run_dir: Path, step: int) -> Path:
    """Return the immutable checkpoint path for a training step."""
    return run_dir / "checkpoints" / f"step_{step:06d}"


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    experiment: ACTExperimentConfig,
) -> None:
    """Write a JSON description of the complete experiment."""
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "arguments": arguments,
        "act_experiment": asdict(experiment),
    }

    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run one complete ACT training experiment."""
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    experiment = ACTExperimentConfig()
    policy_config = experiment.make_policy_config(
        device=args.device,
        pretrained_backbone_weights=None,
    )

    train_dataset = LeRobotDataset(
        repo_id=args.train_repo_id,
        root=args.train_dataset_root,
        delta_timestamps={
            "action": experiment.action_delta_timestamps,
        },
    )
    val_dataset = LeRobotDataset(
        repo_id=args.val_repo_id,
        root=args.val_dataset_root,
        delta_timestamps={
            "action": experiment.action_delta_timestamps,
        },
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=args.device,
        generator=generator,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=args.device,
    )
    reference_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        device=args.device,
    )

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=policy_config,
        dataset_stats=train_dataset.meta.stats,
    )

    policy = ACTPolicy(policy_config).to(args.device)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy_config.optimizer_lr,
        weight_decay=policy_config.optimizer_weight_decay,
    )

    reference_batch = preprocessor(
        next(iter(reference_loader))
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )

    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "checkpoints").mkdir()
    write_run_config(
        args.run_dir / "run_config.json",
        args,
        experiment,
    )

    metrics_path = args.run_dir / "metrics.csv"
    max_val_batches = (
        None if args.val_batches == 0 else args.val_batches
    )

    print("=== ACT experiment ===")
    print("train episodes:", train_dataset.meta.total_episodes)
    print("train frames:", len(train_dataset))
    print("val episodes:", val_dataset.meta.total_episodes)
    print("val frames:", len(val_dataset))
    print("device:", args.device)
    print("steps:", args.steps)
    print("batch size:", args.batch_size)
    print("trainable parameters:", f"{trainable_parameters:,}")
    print("optimizer groups:", len(optimizer.param_groups))
    print("run directory:", args.run_dir)

    best_val_l1 = float("inf")
    best_val_step = 0
    start_time = time.perf_counter()

    policy.train()
    iterator = iter(train_loader)

    total_sum = 0.0
    l1_sum = 0.0
    kld_sum = 0.0
    gradient_sum = 0.0
    window_steps = 0

    with metrics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=METRIC_FIELDS,
        )
        writer.writeheader()
        metrics_file.flush()

        initial_val = evaluate_act_policy(
            policy,
            preprocessor,
            val_loader,
            max_batches=max_val_batches,
        )
        best_val_l1 = initial_val.l1_loss

        writer.writerow(
            {
                "step": 0,
                "elapsed_seconds": 0.0,
                "train_total_loss": "",
                "train_l1_loss": "",
                "train_kld_loss": "",
                "gradient_norm": "",
                "val_l1_loss": initial_val.l1_loss,
            }
        )
        metrics_file.flush()

        print(
            f"step=000000 val_l1={initial_val.l1_loss:.6f} "
            f"val_samples={initial_val.num_samples}"
        )

        for step in range(1, args.steps + 1):
            raw_batch, iterator = next_batch(
                train_loader,
                iterator,
            )
            batch = preprocessor(raw_batch)

            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy(batch)
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=args.max_grad_norm,
            )
            optimizer.step()

            total_sum += float(loss)
            l1_sum += float(loss_dict["l1_loss"])
            kld_sum += float(
                loss_dict.get("kld_loss", 0.0)
            )
            gradient_sum += float(gradient_norm)
            window_steps += 1

            should_validate = step % args.val_every == 0
            should_log = (
                step % args.log_every == 0
                or should_validate
                or step == args.steps
            )

            val_l1: float | str = ""

            if should_validate or step == args.steps:
                validation = evaluate_act_policy(
                    policy,
                    preprocessor,
                    val_loader,
                    max_batches=max_val_batches,
                )
                val_l1 = validation.l1_loss

                if validation.l1_loss < best_val_l1:
                    best_val_l1 = validation.l1_loss
                    best_val_step = step

            if should_log:
                train_total = total_sum / window_steps
                train_l1 = l1_sum / window_steps
                train_kld = kld_sum / window_steps
                mean_gradient = gradient_sum / window_steps
                elapsed = time.perf_counter() - start_time

                writer.writerow(
                    {
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "train_total_loss": train_total,
                        "train_l1_loss": train_l1,
                        "train_kld_loss": train_kld,
                        "gradient_norm": mean_gradient,
                        "val_l1_loss": val_l1,
                    }
                )
                metrics_file.flush()

                message = (
                    f"step={step:06d} "
                    f"train_total={train_total:.6f} "
                    f"train_l1={train_l1:.6f} "
                    f"train_kld={train_kld:.6f} "
                    f"grad={mean_gradient:.3f}"
                )
                if val_l1 != "":
                    message += f" val_l1={val_l1:.6f}"
                print(message, flush=True)

                total_sum = 0.0
                l1_sum = 0.0
                kld_sum = 0.0
                gradient_sum = 0.0
                window_steps = 0

            should_checkpoint = (
                step % args.checkpoint_every == 0
                or step == args.steps
            )

            if should_checkpoint:
                reference = make_reference_chunk(
                    policy,
                    postprocessor,
                    reference_batch,
                )
                destination = checkpoint_path(
                    args.run_dir,
                    step,
                )
                save_act_checkpoint(
                    destination,
                    step=step,
                    policy=policy,
                    optimizer=optimizer,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    scheduler=None,
                    batch_size=args.batch_size,
                    reference_action_chunk=reference,
                )
                print(
                    f"checkpoint saved: {destination}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_time

    print("\n=== Experiment completed ===")
    print("elapsed seconds:", round(elapsed, 2))
    print("best validation L1:", best_val_l1)
    print("best validation step:", best_val_step)
    print("metrics:", metrics_path)
    print("run directory:", args.run_dir)


if __name__ == "__main__":
    main()
