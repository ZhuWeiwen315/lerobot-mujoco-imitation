"""Offline inference-mode validation for ACT policies."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline


@dataclass(frozen=True)
class ACTEvaluationMetrics:
    """Sample-weighted action prediction metrics."""

    l1_loss: float
    num_batches: int
    num_samples: int


@torch.inference_mode()
def evaluate_act_policy(
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    loader: DataLoader,
    *,
    max_batches: int | None = None,
) -> ACTEvaluationMetrics:
    """Measure action L1 loss using ACT's deployment behavior.

    In evaluation mode, ACT does not encode the demonstration actions
    with its VAE encoder. It uses a zero latent, matching normal policy
    inference. Consequently this function reports action L1 only; KL
    divergence is a training-mode regularizer and is intentionally not
    part of this metric.
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None.")

    was_training = policy.training
    policy.eval()

    l1_loss_sum = 0.0
    num_batches = 0
    num_samples = 0

    try:
        for raw_batch in loader:
            if (
                max_batches is not None
                and num_batches >= max_batches
            ):
                break

            batch = preprocessor(raw_batch)
            _, loss_dict = policy(batch)

            if "kld_loss" in loss_dict:
                raise RuntimeError(
                    "ACT unexpectedly returned KL loss in eval mode."
                )

            batch_size = int(batch["action"].shape[0])
            l1_loss_sum += (
                float(loss_dict["l1_loss"]) * batch_size
            )
            num_batches += 1
            num_samples += batch_size
    finally:
        policy.train(was_training)

    if num_samples == 0:
        raise ValueError("Validation loader produced no samples.")

    return ACTEvaluationMetrics(
        l1_loss=l1_loss_sum / num_samples,
        num_batches=num_batches,
        num_samples=num_samples,
    )
