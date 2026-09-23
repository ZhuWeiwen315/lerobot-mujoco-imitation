"""Collect time-aligned raw imitation-learning episodes."""

from pathlib import Path
from typing import Any

import numpy as np

from lerobot_mujoco.envs import PandaLiftEnv
from lerobot_mujoco.experts import PandaLiftFSMExpert


def collect_raw_episode(
    env: PandaLiftEnv,
    expert: PandaLiftFSMExpert,
    seed: int,
) -> dict[str, Any]:
    """Collect one episode as observation_t and action_t pairs."""

    observation, info = env.reset(seed=seed)
    expert.reset()

    images = []
    states = []
    actions = []
    timestamps = []
    frame_indices = []
    fsm_states = []
    cube_positions = []

    for frame_index in range(env.config.max_episode_steps):
        action = expert.act(observation, info)

        # Save observation_t together with the action chosen from it.
        images.append(observation["observation.images.front"].copy())
        states.append(observation["observation.state"].copy())
        actions.append(action.copy())
        timestamps.append(frame_index / env.config.control_frequency)
        frame_indices.append(frame_index)
        fsm_states.append(expert.state.value)
        cube_positions.append(info["cube_pos"].copy())

        observation, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    episode = {
        "images": np.stack(images).astype(np.uint8),
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float32),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fsm_states": np.asarray(fsm_states),
        "cube_positions": np.stack(cube_positions).astype(np.float32),
        "seed": int(seed),
        "success": bool(info["success"]),
        "termination_reason": info["termination_reason"],
        "steps": int(info["step_count"]),
    }

    _validate_episode(episode)
    return episode


def _validate_episode(episode: dict[str, Any]) -> None:
    """Check that all frame-level fields have identical lengths."""

    frame_keys = (
        "images",
        "states",
        "actions",
        "timestamps",
        "frame_indices",
        "fsm_states",
        "cube_positions",
    )
    lengths = {key: len(episode[key]) for key in frame_keys}

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Frame fields are not aligned: {lengths}")

    if lengths["actions"] != episode["steps"]:
        raise ValueError(
            f"Collected {lengths['actions']} actions, "
            f"but the environment reports {episode['steps']} steps."
        )


def save_raw_episode(
    episode: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save one validated episode without Python pickle objects."""

    _validate_episode(episode)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        images=episode["images"],
        states=episode["states"],
        actions=episode["actions"],
        timestamps=episode["timestamps"],
        frame_indices=episode["frame_indices"],
        fsm_states=episode["fsm_states"],
        cube_positions=episode["cube_positions"],
        seed=np.asarray(episode["seed"], dtype=np.int64),
        success=np.asarray(episode["success"], dtype=np.bool_),
        termination_reason=np.asarray(
            str(episode["termination_reason"])
        ),
        steps=np.asarray(episode["steps"], dtype=np.int64),
    )
    return output_path
