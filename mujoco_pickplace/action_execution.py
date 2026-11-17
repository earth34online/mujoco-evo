"""Per-episode execution controllers for the MuJoCo policy client."""

from collections import deque

import numpy as np

from mujoco_pickplace.pick_place_env import PickPlaceEnv


PRECISION_REPLAN_Z = (
    PickPlaceEnv.CUBE_SUPPORT_Z + PickPlaceEnv.EXPERT_GRASP_OFFSET + 0.050
)



class AlignedBinaryGripperEnsemble:
    """Combine up to four predictions for the current control tick."""

    def __init__(self):
        self.predictions = deque(maxlen=4)

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 1 or len(scores) < 4 or not np.isin(scores, (0, 1)).all():
            raise ValueError("Temporal ensemble requires a binary gripper action chunk")
        self.predictions.append(scores.copy())
        aligned = [prior[offset] for offset, prior in enumerate(reversed(self.predictions))]
        weights = np.exp(-0.01 * np.arange(len(aligned)))
        return float(np.dot(aligned[::-1], weights) / weights.sum())


class GripperCommandFilter:
    """Close immediately; debounce only the unsafe open transition."""

    def __init__(
        self,
        required_steps=2,
        close_threshold=0.4,
        open_threshold=0.6,
        initial_command=1.0,
    ):
        if required_steps < 1:
            raise ValueError("required_steps must be at least 1")
        if not 0.0 <= close_threshold < open_threshold <= 1.0:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        self.required_steps = int(required_steps)
        self.close_required_steps = 1
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.command = float(initial_command)
        self.candidate = self.command
        self.candidate_steps = 0

    def update(self, score):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError(f"Non-finite gripper score: {score}")
        if score <= self.close_threshold:
            requested = 0.0
        elif score >= self.open_threshold:
            requested = 1.0
        else:
            requested = self.command

        if requested == self.command:
            self.candidate = self.command
            self.candidate_steps = 0
            return self.command
        if requested != self.candidate:
            self.candidate = requested
            self.candidate_steps = 1
        else:
            self.candidate_steps += 1
        transition_steps = (
            self.close_required_steps if requested < 0.5 else self.required_steps
        )
        if self.candidate_steps >= transition_steps:
            self.command = requested
            self.candidate_steps = 0
        return self.command


class PrecisionExecutionController:
    """Replan grasp-sensitive actions without simulator privileged state.

    The controller only reads the policy-visible 8-D proprioception.  It keeps
    the normal action-chunk horizon for free-space motion and replans every
    step near the grasp plane.  A gripper transition is executed at its
    predicted position in the chunk, then ends that chunk so π-MEM receives a
    fresh visual/proprioceptive observation of the changed gripper state.

    Finger opening is deliberately not treated as grasp confirmation.  A
    corner collision and a stable two-pad grasp can produce nearly identical
    finger qpos, so latching either as success can disable the recovery loop
    precisely after a localization error.
    """

    def __init__(self, precision_z=PRECISION_REPLAN_Z):
        self.precision_z = float(precision_z)

    @staticmethod
    def _validate_robot_state(robot_state):
        robot_state = np.asarray(robot_state, dtype=np.float32)
        if robot_state.shape != (8,):
            raise ValueError(f"Expected 8-D robot_state, got {robot_state.shape}")
        if not np.isfinite(robot_state).all():
            raise ValueError("robot_state contains NaN or Inf")
        return robot_state

    def execution_horizon(
        self,
        action_chunk,
        requested_horizon,
        robot_state,
        gripper_filter,
        enabled=True,
    ):
        robot_state = self._validate_robot_state(robot_state)
        prefix = np.asarray(action_chunk, dtype=np.float32)[:requested_horizon]
        if prefix.ndim != 2 or prefix.shape[1] < 7:
            raise ValueError("action_chunk must have shape [horizon, >=7]")

        if not enabled:
            return int(requested_horizon)

        near_grasp_plane = float(robot_state[2]) <= self.precision_z
        if near_grasp_plane:
            return 1

        # Do not collapse to horizon=1 merely because a later action requests
        # closing: doing so repeatedly discards the transition before it can
        # execute.  Run through the first transition and then replan.  The same
        # boundary makes release debounce meaningful across independent model
        # observations instead of letting two open scores from one stale chunk
        # release an object during transport.
        if gripper_filter.command >= 0.5:
            transition_indices = np.flatnonzero(
                prefix[:, 6] <= gripper_filter.close_threshold
            )
        else:
            transition_indices = np.flatnonzero(
                prefix[:, 6] >= gripper_filter.open_threshold
            )
        if transition_indices.size:
            return int(transition_indices[0]) + 1
        return int(requested_horizon)
