"""Identity and execution defaults for the released MuJoCo checkpoint."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "ckpt/archive/evo_pi-mem_success"
TRAINING_STEP = 40000
WEIGHTS_SHA256 = "97112fd84f1de67192c032a608b6a2d7529ef2a14ed65284c7cb15b9532bfabb"
TASK1_OPEN_THRESHOLD = 0.45
