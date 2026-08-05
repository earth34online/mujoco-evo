import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from pick_place_env import PickPlaceEnv, ScriptedExpertPolicy

print("creating env...")
env = PickPlaceEnv(image_size=64)
def state_only_obs():
    eef = env.data.body("eef").xpos.copy()
    cube = env.data.body("cube").xpos.copy()
    goal = env.model.body("goal").pos.copy()
    return {"state": np.r_[eef, cube, goal, env.gripper].astype(np.float32)}

env.obs = state_only_obs
print("env OK")
ok = 0
for seed in range(6):
    obs = env.reset(seed=seed)
    expert = ScriptedExpertPolicy(env)
    done = False
    steps = 0
    for i in range(140):
        obs, done = env.step(expert(obs))
        if done:
            steps = i + 1
            break
    ok += int(done)
    print(f"seed={seed} success={done} steps={steps}")
env.renderer.close()
print(f"SUCCESS {ok}/6")
