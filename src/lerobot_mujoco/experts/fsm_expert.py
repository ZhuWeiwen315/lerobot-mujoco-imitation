"""Finite-state-machine expert for the Panda Lift task."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ExpertState(str, Enum):
    """Discrete stages of a scripted grasp trajectory."""

    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    HOLD = "hold"
    DONE = "done"
    FAIL = "fail"


@dataclass(frozen=True)
class FSMExpertConfig:
    """Motion parameters expressed in metres and normalized actions."""

    approach_height: float = 0.12
    grasp_height_offset: float = 0.01
    position_tolerance: float = 0.008
    controller_position_scale: float = 0.05
    max_position_action: float = 0.4
    grasp_steps: int = 15
    lift_distance: float = 0.15


class PandaLiftFSMExpert:
    """Closed-loop expert that follows the observed cube position."""

    def __init__(self, config: FSMExpertConfig | None = None) -> None:
        self.config = config or FSMExpertConfig()
        self.state = ExpertState.APPROACH
        self.last_target: np.ndarray | None = None
        self._steps_in_state = 0
        self._lift_target: np.ndarray | None = None

    def reset(self) -> None:
        """Start a new episode in the approach state."""

        self.state = ExpertState.APPROACH
        self.last_target = None
        self._steps_in_state = 0
        self._lift_target = None

    def act(
        self,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> np.ndarray:
        """Return one normalized OSC_POSITION action."""

        eef_pos = np.asarray(
            observation["observation.state"][-3:], dtype=np.float32
        )
        cube_pos = np.asarray(info["cube_pos"], dtype=np.float32)

        approach_target = cube_pos + np.array(
            [0.0, 0.0, self.config.approach_height], dtype=np.float32
        )
        grasp_target = cube_pos + np.array(
            [0.0, 0.0, self.config.grasp_height_offset], dtype=np.float32
        )

        if (
            self.state == ExpertState.APPROACH
            and self._at_target(eef_pos, approach_target)
        ):
            self._transition(ExpertState.DESCEND)

        elif (
            self.state == ExpertState.DESCEND
            and self._at_target(eef_pos, grasp_target)
        ):
            self._transition(ExpertState.GRASP)

        elif (
            self.state == ExpertState.GRASP
            and self._steps_in_state >= self.config.grasp_steps
        ):
            self._lift_target = eef_pos + np.array(
                [0.0, 0.0, self.config.lift_distance], dtype=np.float32
            )
            self._transition(ExpertState.LIFT)

        elif (
            self.state == ExpertState.LIFT
            and bool(info.get("is_lifted", False))
        ):
            # Freeze the current pose so HOLD maintains a fixed height.
            self._lift_target = eef_pos.copy()
            self._transition(ExpertState.HOLD)

        if self.state == ExpertState.APPROACH:
            target, gripper = approach_target, -1.0
        elif self.state == ExpertState.DESCEND:
            target, gripper = grasp_target, -1.0
        elif self.state == ExpertState.GRASP:
            target, gripper = grasp_target, 1.0
        elif self.state in {ExpertState.LIFT, ExpertState.HOLD}:
            if self._lift_target is None:
                raise RuntimeError("Lift target was not initialized.")
            target, gripper = self._lift_target, 1.0
        else:
            target, gripper = eef_pos.copy(), 1.0

        self.last_target = target.copy()
        action = np.zeros(4, dtype=np.float32)

        if self.state not in {ExpertState.DONE, ExpertState.FAIL}:
            error = target - eef_pos
            action[:3] = np.clip(
                error / self.config.controller_position_scale,
                -self.config.max_position_action,
                self.config.max_position_action,
            )

        action[3] = gripper
        self._steps_in_state += 1
        return action

    def _transition(self, new_state: ExpertState) -> None:
        """Enter a new state and reset its local step counter."""

        self.state = new_state
        self._steps_in_state = 0

    def _at_target(
        self,
        position: np.ndarray,
        target: np.ndarray,
    ) -> bool:
        """Check whether the end effector is close enough to a target."""

        error = np.linalg.norm(target - position)
        return bool(error <= self.config.position_tolerance)
