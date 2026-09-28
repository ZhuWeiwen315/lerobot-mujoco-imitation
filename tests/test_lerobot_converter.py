"""Tests for raw NPZ to LeRobotDataset v3 conversion."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_mujoco.datasets import (
    convert_raw_episodes,
    make_lerobot_features,
    validate_raw_episode,
)


def write_raw_episode(
    path: Path,
    *,
    length: int,
    seed: int,
    success: bool = True,
) -> None:
    """Write a small aligned raw episode for testing."""
    rng = np.random.default_rng(seed)

    images = rng.integers(
        0,
        256,
        size=(length, 96, 96, 3),
        dtype=np.uint8,
    )
    states = rng.normal(size=(length, 12)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(length, 4)).astype(np.float32)

    np.savez_compressed(
        path,
        images=images,
        states=states,
        actions=actions,
        timestamps=np.arange(length, dtype=np.float32) / 20,
        frame_indices=np.arange(length, dtype=np.int64),
        success=np.bool_(success),
        steps=np.int64(length),
    )


class LeRobotConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        parent = os.environ.get("LEROBOT_MUJOCO_TEST_TMPDIR")
        self.temporary_directory = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_feature_schema_and_raw_validation(self) -> None:
        raw_path = self.root / "episode_000000.npz"
        write_raw_episode(raw_path, length=4, seed=0)

        self.assertEqual(validate_raw_episode(raw_path), 4)

        features = make_lerobot_features()
        self.assertEqual(
            features["observation.images.front"]["shape"],
            (96, 96, 3),
        )
        self.assertEqual(
            features["observation.state"]["shape"],
            (12,),
        )
        self.assertEqual(features["action"]["shape"], (4,))

    def test_failed_episode_is_rejected(self) -> None:
        raw_path = self.root / "failed_episode.npz"
        write_raw_episode(
            raw_path,
            length=3,
            seed=1,
            success=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Only successful episodes",
        ):
            validate_raw_episode(raw_path)

    def test_convert_and_reload_two_episodes(self) -> None:
        first_path = self.root / "episode_000000.npz"
        second_path = self.root / "episode_000001.npz"
        dataset_root = self.root / "lerobot_dataset"

        write_raw_episode(first_path, length=4, seed=2)
        write_raw_episode(second_path, length=5, seed=3)

        summary = convert_raw_episodes(
            raw_paths=[second_path, first_path],
            dataset_root=dataset_root,
            repo_id="local/panda-lift-converter-test",
            image_writer_threads=0,
        )

        self.assertEqual(summary.num_episodes, 2)
        self.assertEqual(summary.num_frames, 9)

        dataset = LeRobotDataset(
            repo_id="local/panda-lift-converter-test",
            root=dataset_root,
        )

        self.assertEqual(len(dataset), 9)
        self.assertEqual(dataset.meta.total_episodes, 2)

        first_sample = dataset[0]
        second_episode_sample = dataset[4]

        self.assertEqual(int(first_sample["episode_index"]), 0)
        self.assertEqual(int(first_sample["frame_index"]), 0)
        self.assertEqual(int(second_episode_sample["episode_index"]), 1)
        self.assertEqual(int(second_episode_sample["frame_index"]), 0)
        self.assertEqual(
            second_episode_sample["task"],
            "Pick up the cube and lift it.",
        )
        self.assertEqual(
            tuple(first_sample["observation.images.front"].shape),
            (3, 96, 96),
        )

        with np.load(first_path, allow_pickle=False) as raw:
            np.testing.assert_allclose(
                first_sample["observation.state"].numpy(),
                raw["states"][0],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                first_sample["action"].numpy(),
                raw["actions"][0],
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
