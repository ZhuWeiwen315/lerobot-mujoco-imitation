"""Convert aligned raw Panda Lift episodes to LeRobotDataset v3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_TASK = "Pick up the cube and lift it."
DEFAULT_FPS = 20

STATE_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper_left",
    "gripper_right",
    "eef_x",
    "eef_y",
    "eef_z",
]

ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "gripper"]


@dataclass(frozen=True)
class ConversionSummary:
    """Summary of a completed raw-to-LeRobot conversion."""

    root: Path
    num_episodes: int
    num_frames: int


def make_lerobot_features() -> dict:
    """Return the project feature schema for LeRobotDataset v3."""
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": STATE_NAMES.copy(),
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ACTION_NAMES.copy(),
        },
    }


def validate_raw_episode(path: str | Path, fps: int = DEFAULT_FPS) -> int:
    """Validate one successful, time-aligned raw episode.

    Returns:
        Number of behavior-cloning frames in the episode.
    """
    path = Path(path)

    required_keys = {
        "images",
        "states",
        "actions",
        "timestamps",
        "frame_indices",
        "success",
        "steps",
    }

    with np.load(path, allow_pickle=False) as episode:
        missing = required_keys - set(episode.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")

        if not bool(episode["success"].item()):
            raise ValueError(f"Only successful episodes may be converted: {path}")

        length = len(episode["actions"])
        lengths = {
            "images": len(episode["images"]),
            "states": len(episode["states"]),
            "actions": length,
            "timestamps": len(episode["timestamps"]),
            "frame_indices": len(episode["frame_indices"]),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Misaligned episode {path}: {lengths}")

        if int(episode["steps"].item()) != length:
            raise ValueError(
                f"steps does not match frame count in {path}: "
                f"{int(episode['steps'].item())} != {length}"
            )

        expected_shapes = {
            "images": (length, 96, 96, 3),
            "states": (length, 12),
            "actions": (length, 4),
        }
        for key, expected_shape in expected_shapes.items():
            if episode[key].shape != expected_shape:
                raise ValueError(
                    f"Unexpected {key} shape in {path}: "
                    f"{episode[key].shape} != {expected_shape}"
                )

        expected_dtypes = {
            "images": np.dtype(np.uint8),
            "states": np.dtype(np.float32),
            "actions": np.dtype(np.float32),
        }
        for key, expected_dtype in expected_dtypes.items():
            if episode[key].dtype != expected_dtype:
                raise ValueError(
                    f"Unexpected {key} dtype in {path}: "
                    f"{episode[key].dtype} != {expected_dtype}"
                )

        np.testing.assert_array_equal(
            episode["frame_indices"],
            np.arange(length),
        )
        np.testing.assert_allclose(
            episode["timestamps"],
            np.arange(length, dtype=np.float32) / fps,
            atol=1e-6,
        )

    return length


def convert_raw_episodes(
    raw_paths: Iterable[str | Path],
    dataset_root: str | Path,
    repo_id: str,
    *,
    task: str = DEFAULT_TASK,
    fps: int = DEFAULT_FPS,
    robot_type: str = "franka_panda",
    image_writer_threads: int = 4,
) -> ConversionSummary:
    """Convert successful raw NPZ episodes into LeRobotDataset v3."""
    paths = sorted(Path(path) for path in raw_paths)
    root = Path(dataset_root)

    if not paths:
        raise ValueError("At least one raw episode is required.")
    if root.exists():
        raise FileExistsError(f"Dataset root already exists: {root}")

    # Validate every input before creating any output files.
    episode_lengths = [
        validate_raw_episode(path, fps=fps)
        for path in paths
    ]

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=robot_type,
        features=make_lerobot_features(),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
    )

    try:
        for path, length in zip(paths, episode_lengths, strict=True):
            with np.load(path, allow_pickle=False) as episode:
                for frame_index in range(length):
                    dataset.add_frame(
                        {
                            "observation.images.front": episode["images"][
                                frame_index
                            ],
                            "observation.state": episode["states"][frame_index],
                            "action": episode["actions"][frame_index],
                            "task": task,
                        }
                    )

            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()

    return ConversionSummary(
        root=root,
        num_episodes=len(paths),
        num_frames=sum(episode_lengths),
    )
