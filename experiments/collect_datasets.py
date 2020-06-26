import os, sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

import uuid 
import pickle as pkl

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/gym_thing/gym_thing/")
from visual_pusher_env import ThingVisualPusher
import pybullet as p

def write_file_pkl(data, name, location="."):
    tag = uuid.uuid4().hex[:].upper()
    filename = os.path.join(location, name + "_" + tag)
    os.makedirs(location, exist_ok=True)
    with open(f'{filename}.pkl', 'wb') as f:
        pkl.dump(data, f)

def visual_haptic_1D():
    n_steps = 32
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False, substeps=n_steps, frame_skip=2)
    config = env.get_config()

    s_list = list(np.arange(1.0, 2.0, 0.01))  
    n = len(s_list)
    ll = 30
      
    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
        "config": config
    }

    for ii, s in enumerate(s_list):
        env.reset()
        for jj in range(ll): 
            obs, reward, done, info = env.step(action=np.array([0.20 * s, 0]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]

    return data

if __name__ == "__main__":
    data = visual_haptic_1D()

    write_file_pkl(data=data, name="visual_haptic_1d", location="./data/datasets/")