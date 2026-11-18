"""Regression coverage for the released entry points and gripper timing."""

from dataclasses import replace

import pytest

from mujoco_pickplace import release, tasks
from mujoco_pickplace.action_execution import AlignedBinaryGripperEnsemble, GripperCommandFilter
from mujoco_pickplace.eval_policy_client import parse_args, resolve_execution_policy
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


def test_release_reobserves_each_action_and_preserves_explicit_overrides():
    metadata = {"recommended_execution_horizon": 4,
                "checkpoint_identity": {"weights_sha256": release.WEIGHTS_SHA256}}
    assert resolve_execution_policy(parse_args([]), metadata)[0] == 1
    assert resolve_execution_policy(parse_args(["--task", "1", "--horizon", "4"]), metadata)[0] == 4
    metadata["checkpoint_identity"]["weights_sha256"] = "other-checkpoint"
    assert resolve_execution_policy(parse_args([]), metadata)[0] == 4


def _recorded_native_contract(source):
    import hashlib
    import json
    sources = {"task_env.py": source, "Task.prepare_action": "a" * 64}
    return dict(schema="mujoco-task-environment", version=2, task_id=4,
                control_period_seconds=0.1, action_convention="world_delta_pose",
                state_convention="hand_axis_angle", collection_config={},
                assets_sha256={"scene.xml": "b" * 64}, quality_schema={"version": 3},
                physics_engine=dict(name="mujoco", python_package_version="3.11.0", runtime_version="3.11.0"),
                controller_sources_sha256=sources,
                controller_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest())


def test_reviewed_controller_refinement_keeps_training_provenance():
    before = _recorded_native_contract(release.DRAWER_CONTROLLER_BEFORE)
    after = _recorded_native_contract(release.DRAWER_CONTROLLER_AFTER)
    checked = release.compare_evaluation_environment(before, after)
    assert checked["status"] == "verified"
    assert checked["runtime_refinement"]["physics_substeps"] == 2
    assert checked["differences"]
    assert before["controller_sources_sha256"]["task_env.py"] == release.DRAWER_CONTROLLER_BEFORE


@pytest.mark.parametrize("change", ["assets", "period", "engine", "other_controller", "unknown_source"])
def test_controller_refinement_does_not_hide_other_environment_changes(change):
    before = _recorded_native_contract(release.DRAWER_CONTROLLER_BEFORE)
    after = _recorded_native_contract(release.DRAWER_CONTROLLER_AFTER)
    if change == "assets":
        after["assets_sha256"]["scene.xml"] = "c" * 64
    elif change == "period":
        after["control_period_seconds"] = 0.2
    elif change == "engine":
        after["physics_engine"].update(python_package_version="different", runtime_version="different")
    elif change == "other_controller":
        after["controller_sources_sha256"]["Task.prepare_action"] = "c" * 64
    else:
        after["controller_sources_sha256"]["task_env.py"] = "c" * 64
    assert release.compare_evaluation_environment(before, after)["status"] == "mismatch"


def test_drawer_physics_tick_preserves_time_and_restores_options(monkeypatch):
    from types import SimpleNamespace
    from mujoco_pickplace.task_env import NativePanda
    import mujoco
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
    data = SimpleNamespace(time=0.0)
    env = SimpleNamespace(model=model, data=data, task=4, DRAWER_PHYSICS_SUBSTEPS=2)
    observed = []
    def step(model, data, nstep=1):
        observed.append(model.opt.timestep)
        data.time += model.opt.timestep * nstep
    monkeypatch.setattr(mujoco, "mj_step", step)
    NativePanda._physics_tick(env)
    assert data.time == pytest.approx(0.002)
    assert observed == [0.001]
    assert model.opt.timestep == 0.002
    env.task = 3
    NativePanda._physics_tick(env)
    assert observed[-1] == 0.002
    env.task = 4
    def fail(*args, **kwargs):
        raise RuntimeError("physics failure")
    monkeypatch.setattr(mujoco, "mj_step", fail)
    with pytest.raises(RuntimeError, match="physics failure"):
        NativePanda._physics_tick(env)
    assert model.opt.timestep == 0.002


def test_shared_server_execution_recommendation_preserves_other_checkpoints():
    assert release.recommended_execution_horizon({"weights_sha256": release.WEIGHTS_SHA256}, 4) == 1
    assert release.recommended_execution_horizon({"weights_sha256": "other"}, 14) == 14
    assert release.recommended_execution_horizon(None, 4) == 4
