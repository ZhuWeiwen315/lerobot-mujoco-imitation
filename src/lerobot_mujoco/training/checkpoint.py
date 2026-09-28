"""Checkpoint persistence for ACT policies and processor pipelines."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.common.train_utils import (
    load_training_state,
    save_training_state,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import PolicyProcessorPipeline


def save_act_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    scheduler: Any = None,
    batch_size: int | None = None,
    reference_action_chunk: np.ndarray | None = None,
) -> Path:
    """Save an ACT policy and all state needed to resume training."""
    checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_dir}"
        )
    if step < 0:
        raise ValueError("step must be non-negative.")

    pretrained_dir = checkpoint_dir / "pretrained_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    policy.save_pretrained(
        pretrained_dir,
        push_to_hub=False,
    )
    preprocessor.save_pretrained(
        pretrained_dir,
        push_to_hub=False,
    )
    postprocessor.save_pretrained(
        pretrained_dir,
        push_to_hub=False,
    )

    save_training_state(
        checkpoint_dir=checkpoint_dir,
        train_step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        num_processes=1,
        batch_size=batch_size,
    )

    if reference_action_chunk is not None:
        np.save(
            checkpoint_dir / "reference_action_chunk.npy",
            reference_action_chunk,
        )

    return checkpoint_dir


def load_act_policy_bundle(
    checkpoint_dir: str | Path,
    *,
    device: str,
) -> tuple[
    ACTPolicy,
    PolicyProcessorPipeline,
    PolicyProcessorPipeline,
]:
    """Load an ACT policy together with its preprocessing pipelines."""
    checkpoint_dir = Path(checkpoint_dir)
    pretrained_dir = checkpoint_dir / "pretrained_model"

    if not pretrained_dir.is_dir():
        raise FileNotFoundError(
            f"Missing pretrained model directory: {pretrained_dir}"
        )

    policy = ACTPolicy.from_pretrained(
        pretrained_dir,
        local_files_only=True,
    ).to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_dir,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )

    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_dir,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    return policy, preprocessor, postprocessor


def restore_act_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
) -> tuple[int, torch.optim.Optimizer, Any]:
    """Restore the step, optimizer, scheduler, and random-number state."""
    return load_training_state(
        checkpoint_dir=Path(checkpoint_dir),
        optimizer=optimizer,
        scheduler=scheduler,
    )
