"""Regression coverage for the released entry points and gripper timing."""

from dataclasses import replace

import pytest

from mujoco_pickplace import release, tasks
from mujoco_pickplace.action_execution import AlignedBinaryGripperEnsemble, GripperCommandFilter
from mujoco_pickplace.eval_policy_client import parse_args
from mujoco_pickplace.training_config import parse_training_args


def test_default_evaluation_targets_the_release_without_a_fixed_seed():
    args = parse_args([])
    assert args.expected_checkpoint == release.CHECKPOINT
    assert args.expected_step == release.TRAINING_STEP
    assert args.tasks == [1, 2, 3]
    assert args.start_seed is None
    assert args.task3_gripper_temporal_ensemble is None
    assert not parse_args(["--no-task3-gripper-temporal-ensemble"]).task3_gripper_temporal_ensemble


def test_gripper_ensemble_uses_predictions_for_the_same_control_tick():
    ensemble = AlignedBinaryGripperEnsemble()
    grip = GripperCommandFilter(required_steps=1, initial_command=0.0)
    assert grip.update(ensemble.update([0, 1, 1, 1])) == 0.0
    assert grip.update(ensemble.update([0, 1, 1, 1])) == 0.0
    assert grip.update(ensemble.update([0, 1, 1, 1])) == 1.0
    assert AlignedBinaryGripperEnsemble().update([0, 1, 1, 1]) == 0.0


def test_training_profile_preserves_legacy_defaults_and_explicit_overrides():
    assert parse_training_args([]).action_head == "flowmatching"
    args = parse_training_args(["--pi-mem", "--max_steps", "12"])
    assert args.action_head == "direct"
    assert args.task4_binary_gripper_supervision is True
    assert args.resume and args.resume_pretrain
    assert args.resume_path == str(release.CHECKPOINT)
    assert args.max_steps == 12
    assert args.seed is None


@pytest.mark.skipif(
    not (release.PROJECT_ROOT / "mujoco_menagerie").is_dir(),
    reason="The MuJoCo Menagerie Panda assets are not installed",
)
def test_asset_fingerprints_are_the_same_for_a_linked_checkout(tmp_path, monkeypatch):
    root = release.PROJECT_ROOT
    for directory in ("mujoco_pickplace", "mujoco_menagerie"):
        (tmp_path / directory).symlink_to(root / directory, target_is_directory=True)
    for number in (1, 3, 4):
        task = tasks.get_task(number)
        before = tasks.task_environment_contract(task)
        linked = replace(task, scene=tmp_path / task.scene.relative_to(root))
        with monkeypatch.context() as patch:
            patch.setattr(tasks, "ROOT", tmp_path)
            assert tasks.task_environment_contract(linked) == before
