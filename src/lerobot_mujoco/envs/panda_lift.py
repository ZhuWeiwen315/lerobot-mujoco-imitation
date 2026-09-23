"""Franka Panda Lift environment adapter for robosuite."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import robosuite
from robosuite.controllers import (
    load_composite_controller_config,
    load_part_controller_config,
)


@dataclass(frozen=True)
class PandaLiftConfig:
    """Configuration for the first Panda Lift environment."""

    camera_name: str = "sideview"
    image_size: int = 96
    control_frequency: int = 20
    max_episode_steps: int = 300
    lift_height_margin: float = 0.08
    success_hold_steps: int = 10


def make_osc_position_controller_config() -> dict[str, Any]:
    """Build a Panda controller with XYZ position and gripper actions."""

    config = load_composite_controller_config(controller="BASIC")
    gripper_config = config["body_parts"]["right"]["gripper"]

    config["body_parts"]["right"] = load_part_controller_config(
        default_controller="OSC_POSITION"
    )
    config["body_parts"]["right"]["gripper"] = gripper_config
    return config


class PandaLiftEnv:
    """Small Gymnasium-style adapter around robosuite Lift."""

    def __init__(self, config: PandaLiftConfig | None = None) -> None:
        self.config = config or PandaLiftConfig()
        self._env = None
        self._seed: int | None = None
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False

    def _make_env(self, seed: int | None) -> None:
        """Create the underlying robosuite environment."""

        self._env = robosuite.make(
            "Lift",
            robots=["Panda"],
            controller_configs=make_osc_position_controller_config(),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=self.config.camera_name,
            camera_heights=self.config.image_size,
            camera_widths=self.config.image_size,
            control_freq=self.config.control_frequency,
            horizon=self.config.max_episode_steps,
            ignore_done=False,
            hard_reset=True,
            reward_shaping=False,
            seed=seed,
        )
        self._seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the task and return policy observations plus debug info."""

        if seed is not None:
            self.close()
            self._make_env(seed)
        elif self._env is None:
            self._make_env(seed=None)

        raw_observation = self._env.reset()
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False

        observation = self._format_observation(raw_observation)
        info = self._make_info(raw_observation)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one action using Gymnasium terminated / truncated semantics."""

        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), received {action.shape}.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity.")

        clipped_action = np.clip(action, -1.0, 1.0)
        raw_observation, robosuite_reward, robosuite_done, _ = self._env.step(
            clipped_action
        )
        self._step_count += 1

        cube_height = float(raw_observation["cube_pos"][2])
        table_height = float(self._env.model.mujoco_arena.table_offset[2])
        is_lifted = (
            cube_height
            > table_height + self.config.lift_height_margin
        )

        if is_lifted:
            self._success_streak += 1
        else:
            self._success_streak = 0

        terminated = self._success_streak >= self.config.success_hold_steps
        truncated = (
            not terminated
            and (
                self._step_count >= self.config.max_episode_steps
                or bool(robosuite_done)
            )
        )

        if terminated:
            termination_reason = "success"
        elif truncated:
            termination_reason = "max_episode_steps"
        else:
            termination_reason = None

        self._episode_done = terminated or truncated
        observation = self._format_observation(raw_observation)
        info = self._make_info(raw_observation)
        info.update(
            {
                "step_count": self._step_count,
                "is_lifted": is_lifted,
                "success_streak": self._success_streak,
                "success": terminated,
                "termination_reason": termination_reason,
                "action_was_clipped": not np.array_equal(
                    action, clipped_action
                ),
                "applied_action": clipped_action.copy(),
                "robosuite_reward": float(robosuite_reward),
            }
        )

        reward = float(terminated)
        return observation, reward, terminated, truncated, info

    def _format_observation(
        self,
        raw_observation: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Select the information that the learned policy may observe."""

        state = np.concatenate(
            [
                raw_observation["robot0_joint_pos"],
                raw_observation["robot0_gripper_qpos"],
                raw_observation["robot0_eef_pos"],
            ]
        ).astype(np.float32)

        image_key = f"{self.config.camera_name}_image"
        image = np.asarray(raw_observation[image_key], dtype=np.uint8).copy()

        return {
            "observation.images.front": image,
            "observation.state": state,
        }

    def _make_info(
        self,
        raw_observation: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Return privileged values for experts, evaluation, and debugging."""

        return {
            "seed": self._seed,
            "cube_pos": raw_observation["cube_pos"].astype(np.float32).copy(),
            "eef_pos": raw_observation["robot0_eef_pos"].astype(np.float32).copy(),
        }

    def close(self) -> None:
        """Release the MuJoCo environment and rendering context."""

        if self._env is not None:
            self._env.close()
            self._env = None
