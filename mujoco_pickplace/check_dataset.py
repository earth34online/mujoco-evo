import sys
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVO_ROOT = PROJECT_ROOT / "Evo-1" / "Evo_1"
CONFIG_PATH = EVO_ROOT / "dataset" / "config.yaml"
CACHE_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "cache" / "mujoco_pickplace"

sys.path.append(str(EVO_ROOT))

from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = LeRobotDataset(
        config=config,
        image_size=448,
        action_horizon=50,
        max_samples_per_file=None,
        cache_dir=CACHE_DIR,
        use_augmentation=False,
    )

    print("dataset length:", len(dataset))
    item = dataset[0]

    for key, value in item.items():
        if hasattr(value, "shape"):
            print(key, value.shape, value.dtype)
        else:
            print(key, value)


if __name__ == "__main__":
    main()


