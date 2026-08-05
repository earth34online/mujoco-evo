import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import imageio.v2 as imageio
from pick_place_env import PickPlaceEnv

env = PickPlaceEnv()
obs = env.reset(seed=0)
# save front view at the HOME pose (arm upright, fingers down, gripper visible)
img = env.render("front")
imageio.imwrite("outputs/_gripper_front.png", img)
print("saved front")

# check world positions of the key gripper parts
eef = env.data.body("eef").xpos
R = env.data.xmat[env.model.body("eef").id].reshape(3, 3)
flange_world = eef + R @ np.array([0, 0, 0.010])     # top of flange
palm_world = eef + R @ np.array([0, 0, -0.030])       # palm center
lf = env.data.body("left_finger_body").xpos
rf = env.data.body("right_finger_body").xpos
print(f"eef world z = {eef[2]:.3f}")
print(f"flange (top, local +0.01) z = {flange_world[2]:.3f}")
print(f"palm center z = {palm_world[2]:.3f}")
print(f"left_finger z = {lf[2]:.3f}, right_finger z = {rf[2]:.3f}")
print(f"eef local z-axis world = {np.round(R[:, 2], 3)}")
print("=> 从上到下: flange, palm, fingers", "正确" if lf[2] < palm_world[2] < flange_world[2] else "有误")
