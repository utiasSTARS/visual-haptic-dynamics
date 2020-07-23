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

def visual_haptic_2D_osc():
    n_steps = 32
    env = ThingVisualPusher(
        is_render=True,
        render_w=64, 
        render_h=64, 
        goal_vis=False, 
        substeps=n_steps, 
        frame_skip=2
    )
    config = env.get_config()

    repeat = 3

    mag_list = list(np.arange(1.0, 1.96, 0.03)) * repeat
    n_mag = len(mag_list)

    n = n_mag
    ll = 16

    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
        "action": np.zeros((n, ll, 2)), 
        "gt_plate_pos": np.zeros((n, ll, 3))
        "config": config
    }
    
    for ii, m in enumerate(mag_list):
        env.reset()
        for jj in range(ll): 
            u_x = 0.30 * m
            u_y = np.clip(np.random.normal(0, 0.25), -0.30, 0.30) 
            obs, reward, done, info = env.step(action=np.array([u_x, u_y]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]
            data["action"][ii, jj] = np.array([u_x, u_y])
            data["gt_plate_pos"][ii, jj] = info["achieved_goal"] 

    return data

def visual_haptic_2D():
    n_steps = 32
    env = ThingVisualPusher(
        is_render=False,
        render_w=64, 
        render_h=64, 
        goal_vis=False, 
        substeps=n_steps, 
        frame_skip=2
    )
    config = env.get_config()

    mag_list = list(np.arange(1.0, 1.96, 0.03))  
    n_mag = len(mag_list)

    init_pos_list = list(3 * np.arange(-0.050, 0.060, 0.005))
    n_init = len(init_pos_list)
    
    n = n_init * n_mag
    ll = 10

    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
        "action": np.zeros((n, ll, 2)), 
        "gt_plate_pos": np.zeros((n, ll, 3))
        "config": config
    }
    
    for ii, h in enumerate(init_pos_list):
        for m in mag_list:
            env.reset()
            for _ in range(5):
                env.step(action=np.array([0, h]))
            for jj in range(ll): 
                u = np.array([0.50 * m, 0])
                obs, reward, done, info = env.step(action=u)
                data["img"][ii, jj] = obs["img"]
                data["ft"][ii, jj] = obs["ft"]
                data["arm"][ii, jj] = obs["arm"]
                data["action"][ii, jj] = u
                data["gt_plate_pos"][ii, jj] = info["achieved_goal"] 

    return data

def visual_haptic_1D():
    n_steps = 32
    env = ThingVisualPusher(
        render_w=64, 
        render_h=64, 
        goal_vis=False, 
        substeps=n_steps, 
        frame_skip=2
    )
    config = env.get_config()

    s_list = list(np.arange(1.0, 1.96, 0.01))  
    n = len(s_list)
    ll = 10
      
    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
        "action": np.zeros((n, ll, 2)), 
        "gt_plate_pos": np.zeros((n, ll, 3))
        "config": config
    }

    for ii, s in enumerate(s_list):
        env.reset()
        for jj in range(ll): 
            u = np.array([0.50 * s, 0])
            obs, reward, done, info = env.step(action=u)
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]
            data["action"][ii, jj] = u
            data["gt_plate_pos"][ii, jj] = info["achieved_goal"] 

    return data

if __name__ == "__main__":
    # data = visual_haptic_1D()
    # data = visual_haptic_2D()
    data = visual_haptic_2D_osc()

    write_file_pkl(data=data, name="visual_haptic_2D_osc", location="./data/datasets/")