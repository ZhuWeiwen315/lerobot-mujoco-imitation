"""ACT policy configuration for Panda Lift imitation learning."""

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass(frozen=True)
class ACTExperimentConfig:
    """Project-level ACT dimensions and temporal settings."""

    fps: int = 20
    image_key: str = "observation.images.front"
    image_channels: int = 3
    image_height: int = 96
    image_width: int = 96
    state_dim: int = 12
    action_dim: int = 4
    chunk_size: int = 20
    n_action_steps: int = 20

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                "n_action_steps must be between 1 and chunk_size."
            )

    @property
    def action_delta_timestamps(self) -> list[float]:
        """Future action timestamps relative to the current observation."""
        return [
            index / self.fps
            for index in range(self.chunk_size)
        ]

    def make_policy_config(
        self,
        *,
        device: str,
        pretrained_backbone_weights: str | None = None,
    ) -> ACTConfig:
        """Build the LeRobot ACT configuration used by this project."""
        input_features = {
            self.image_key: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(
                    self.image_channels,
                    self.image_height,
                    self.image_width,
                ),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim,),
            ),
        }
        output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            ),
        }

        return ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
            pretrained_backbone_weights=pretrained_backbone_weights,
            push_to_hub=False,
        )
