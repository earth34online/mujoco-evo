import sys
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVO_ROOT = PROJECT_ROOT / "Evo-1" / "Evo_1"
CONFIG_PATH = EVO_ROOT / "dataset" / "config.yaml"

sys.path.append(str(EVO_ROOT))

from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = LeRobotDataset(
        config=config,
        image_size=448,
        action_horizon=14,
        max_samples_per_file=None,
        use_augmentation=False,
        overwrite_horizon_cache=False,
    )

    print("dataset length:", len(dataset))
    item = dataset[0]

    for key, value in item.items():
        if hasattr(value, "shape"):
            print(key, value.shape, value.dtype)
        else:
            print(key, value)
            
    assert item["images"].shape == (3, 3, 448, 448)
    assert item["state"].shape == (24,)
    assert item["action"].shape == (14, 24)
    assert item["image_mask"].tolist() == [True, False, False]
    assert item["action_mask"].shape == (14, 24)

    expected_action_mask = [
        True, True, True,
        False, False, False,
        True,
    ] + [False] * 17

    assert item["action_mask"][0].tolist() == expected_action_mask

    print("[PASS] dataset interface is correct")

if __name__ == "__main__":
    main()


