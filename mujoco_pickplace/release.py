"""Identity and execution defaults for the released MuJoCo checkpoint."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "ckpt/archive/evo_pi-mem_success"
TRAINING_STEP = 40000
WEIGHTS_SHA256 = "97112fd84f1de67192c032a608b6a2d7529ef2a14ed65284c7cb15b9532bfabb"
TASK1_OPEN_THRESHOLD = 0.45
EXECUTION_HORIZON = 1


def recommended_execution_horizon(checkpoint_identity, fallback):
    """Share the released execution default between server and client."""
    if (checkpoint_identity or {}).get("weights_sha256") == WEIGHTS_SHA256:
        return EXECUTION_HORIZON
    return fallback


# Explicit source transition, not a replacement for recorded training provenance.
DRAWER_CONTROLLER_BEFORE = "243c0f5c52ea74cebcb5c5fec251daa8c397bb058ba9626e37ad2938158fdd55"
DRAWER_CONTROLLER_AFTER = "02fb52dc26daa70d5e53ef49464f112bc705d495f097c008a041d1b4dc874397"


def compare_evaluation_environment(trained, current):
    """Recognize the reviewed numerical refinement; reject any other mismatch."""
    import copy
    import hashlib
    import json
    from .episode_dataset import compare_environment_contract

    checked = compare_environment_contract(trained, current)
    if checked["status"] != "mismatch" or current.get("task_id") not in (3, 4):
        return checked
    before = (trained or {}).get("controller_sources_sha256", {})
    after = current.get("controller_sources_sha256", {})
    if (before.get("task_env.py") != DRAWER_CONTROLLER_BEFORE
            or after.get("task_env.py") != DRAWER_CONTROLLER_AFTER):
        return checked
    previous = copy.deepcopy(current)
    previous["controller_sources_sha256"]["task_env.py"] = DRAWER_CONTROLLER_BEFORE
    previous["controller_sha256"] = hashlib.sha256(
        json.dumps(previous["controller_sources_sha256"], sort_keys=True).encode()
    ).hexdigest()
    if compare_environment_contract(trained, previous)["status"] != "verified":
        return checked
    return dict(
        status="verified",
        differences=checked["differences"],
        runtime_refinement=dict(
            name="panda_drawer_half_step_v1",
            training_controller_sha256=DRAWER_CONTROLLER_BEFORE,
            runtime_controller_sha256=DRAWER_CONTROLLER_AFTER,
            physics_substeps=2 if current["task_id"] == 4 else 1,
            control_period_seconds=current["control_period_seconds"],
            reason="Drawer-only numerical integration refinement; other native task unchanged",
        ),
    )
