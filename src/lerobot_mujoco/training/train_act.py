"""Command-line ACT trainer for the Panda Lift dataset."""

import argparse
import random
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ACT on a local Panda Lift LeRobotDataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local LeRobotDataset v3 directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="New output checkpoint directory.",
    )
    parser.add_argument(
        "--repo-id",
        default="zhuweiwen/panda-lift-local",
        help="Dataset identifier used by LeRobot.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of optimizer updates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_batch(
    loader: DataLoader,
    iterator,
):
    """Read a batch and restart the loader after the end of an epoch."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    """Train ACT and write one resumable checkpoint."""
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive.")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {args.dataset_root}"
        )
    if args.checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {args.checkpoint_dir}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seed_everything(args.seed)

    experiment = ACTExperimentConfig()
    policy_config = experiment.make_policy_config(
        device=args.device,
        pretrained_backbone_weights=None,
    )

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        delta_timestamps={
            "action": experiment.action_delta_timestamps,
        },
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=generator,
    )

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=policy_config,
        dataset_stats=dataset.meta.stats,
    )

    policy = ACTPolicy(policy_config).to(args.device)

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy_config.optimizer_lr,
        weight_decay=policy_config.optimizer_weight_decay,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )

    print("=== ACT training ===")
    print("dataset:", args.dataset_root)
    print("frames:", len(dataset))
    print("episodes:", dataset.meta.total_episodes)
    print("device:", args.device)
    print("steps:", args.steps)
    print("batch size:", args.batch_size)
    print("trainable parameters:", f"{trainable_parameters:,}")
    print("optimizer parameter groups:", len(optimizer.param_groups))

    policy.train()
    iterator = iter(loader)

    for step in range(1, args.steps + 1):
        raw_batch, iterator = next_batch(loader, iterator)
        batch = preprocessor(raw_batch)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = policy(batch)
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=args.max_grad_norm,
        )
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:06d} "
                f"loss={float(loss):.6f} "
                f"l1={loss_dict['l1_loss']:.6f} "
                f"kld={loss_dict.get('kld_loss', 0.0):.6f} "
                f"grad_norm={float(gradient_norm):.6f}"
            )

    # A fixed, non-shuffled batch gives us a reproducibility reference.
    reference_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    reference_batch = preprocessor(next(iter(reference_loader)))

    policy.eval()
    policy.reset()

    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(reference_batch)
        environment_chunk = postprocessor(normalized_chunk)

    reference = environment_chunk.detach().cpu().numpy()

    save_act_checkpoint(
        args.checkpoint_dir,
        step=args.steps,
        policy=policy,
        optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        scheduler=None,
        batch_size=args.batch_size,
        reference_action_chunk=reference,
    )

    print("\n=== Training completed ===")
    print("checkpoint:", args.checkpoint_dir)
    print("reference action chunk:", reference.shape)


if __name__ == "__main__":
    main()
