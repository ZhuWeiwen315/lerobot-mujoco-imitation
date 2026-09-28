"""Tests for ACT training configuration and checkpoint validation."""

import tempfile
import unittest
from pathlib import Path

from lerobot.configs.types import FeatureType

from lerobot_mujoco.training import (
    ACTExperimentConfig,
    load_act_policy_bundle,
)


class ACTExperimentConfigTest(unittest.TestCase):
    """Validate project-level ACT dimensions and timing."""

    def test_default_temporal_configuration(self) -> None:
        experiment = ACTExperimentConfig()

        self.assertEqual(experiment.fps, 20)
        self.assertEqual(experiment.chunk_size, 20)
        self.assertEqual(experiment.n_action_steps, 20)
        self.assertEqual(
            len(experiment.action_delta_timestamps),
            20,
        )
        self.assertAlmostEqual(
            experiment.action_delta_timestamps[0],
            0.0,
        )
        self.assertAlmostEqual(
            experiment.action_delta_timestamps[-1],
            0.95,
        )

    def test_policy_feature_schema(self) -> None:
        experiment = ACTExperimentConfig()
        config = experiment.make_policy_config(
            device="cpu",
            pretrained_backbone_weights=None,
        )

        image = config.input_features[
            "observation.images.front"
        ]
        state = config.input_features["observation.state"]
        action = config.output_features["action"]

        self.assertEqual(image.type, FeatureType.VISUAL)
        self.assertEqual(image.shape, (3, 96, 96))
        self.assertEqual(state.type, FeatureType.STATE)
        self.assertEqual(state.shape, (12,))
        self.assertEqual(action.type, FeatureType.ACTION)
        self.assertEqual(action.shape, (4,))
        self.assertEqual(config.chunk_size, 20)

    def test_invalid_temporal_settings_are_rejected(self) -> None:
        invalid_arguments = [
            {"fps": 0},
            {"chunk_size": 0},
            {"n_action_steps": 0},
            {"chunk_size": 20, "n_action_steps": 21},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ACTExperimentConfig(**arguments)


class ACTCheckpointValidationTest(unittest.TestCase):
    """Validate checkpoint failures before loading large models."""

    def test_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-checkpoint"

            with self.assertRaises(FileNotFoundError):
                load_act_policy_bundle(
                    missing,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
