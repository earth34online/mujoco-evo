"""Check the three-task release targets from a completed holdout evaluation."""

import argparse
import json
from pathlib import Path


def check_release(summary, checkpoint, step, start_seed, episodes):
    expected = set(range(start_seed, start_seed + episodes))
    seen = {1: set(), 2: set(), 3: set()}
    successes = dict.fromkeys(seen, 0)
    budgets = {1: 200, 2: 250, 3: 600}
    if summary["status"] != "completed" or summary["tasks"] != [1, 2, 3]:
        raise ValueError("A completed evaluation of Tasks 1, 2 and 3 is required")
    hashes = set()
    for row in summary["episodes"]:
        task, seed = row["task"], row["seed"]
        if task not in seen or seed not in expected or seed in seen[task]:
            raise ValueError(f"Unexpected or repeated task/seed: {task}/{seed}")
        seen[task].add(seed)
        identity = row["checkpoint_metadata"].get("checkpoint_identity") or {}
        if (identity.get("step") != step or not identity.get("path")
                or Path(identity["path"]).resolve() != checkpoint):
            raise ValueError(f"Wrong checkpoint in Task{task}, seed {seed}")
        digest = identity.get("weights_sha256")
        if not digest or row["environment_compatibility"]["status"] != "verified":
            raise ValueError(f"Unverified weights or environment in Task{task}, seed {seed}")
        hashes.add(digest)
        if type(row["success"]) is not bool or not 0 < row["steps"] <= budgets[task]:
            raise ValueError(f"Invalid success flag or execution budget in Task{task}, seed {seed}")
        successes[task] += int(row["success"])
    if len(hashes) != 1 or any(seeds != expected for seeds in seen.values()):
        raise ValueError("Evaluation is incomplete or contains mixed weights")
    passed = successes[1] == episodes and successes[2] == episodes and successes[3] * 100 >= 85 * episodes
    return successes, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=30)
    args = parser.parse_args()
    if args.num_episodes < 1 or args.start_seed < 0 or args.step < 0:
        parser.error("Invalid episode count, seed, or checkpoint step")
    try:
        result, passed = check_release(
            json.loads(args.summary.read_text(encoding="utf-8")), args.checkpoint.resolve(),
            args.step, args.start_seed, args.num_episodes,
        )
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    for task, count in result.items():
        print(f"Task{task}: {count}/{args.num_episodes}")
    print("Release success-rate targets passed" if passed else "Release success-rate targets not met")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
