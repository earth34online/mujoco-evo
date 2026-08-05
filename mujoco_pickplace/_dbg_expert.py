import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from pick_place_env import PickPlaceEnv, scripted_expert

env = PickPlaceEnv()
obs = env.reset(seed=0)
print("initial cube=", np.round(obs["state"][3:6], 3), "goal=", np.round(obs["state"][6:9], 3))
gp = obs["state"][9]
for i in range(30):
    st = obs["state"]
    eef, cube = st[:3], st[3:6]
    if i % 5 == 0:
        print(f"step{i}: eef=({eef[0]:.2f},{eef[1]:.2f},{eef[2]:.2f}) "
              f"cube=({cube[0]:.2f},{cube[1]:.2f},{cube[2]:.2f}) dist={np.linalg.norm(eef-cube):.3f} grip={st[9]:.1f}")
    obs, done = env.step(scripted_expert(obs))
    if done:
        print(f"SUCCESS at {i+1}")
        break
else:
    st = obs["state"]
    cube, goal = st[3:6], st[6:9]
    print(f"FAIL: cube={np.round(cube,3)} goal={np.round(goal,3)} "
          f"dxy={np.linalg.norm(cube[:2]-goal[:2]):.3f} z={cube[2]:.3f}")
