import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from pick_place_env import PickPlaceEnv, scripted_expert

print("creating env...")
env = PickPlaceEnv()
print("env OK")
ok = 0
for seed in range(4):
    obs = env.reset(seed=seed)
    done = False
    steps = 0
    for i in range(80):
        obs, done = env.step(scripted_expert(obs))
        if done:
            steps = i + 1
            break
    ok += int(done)
    print(f"seed={seed} success={done} steps={steps}")
print(f"SUCCESS {ok}/4")
