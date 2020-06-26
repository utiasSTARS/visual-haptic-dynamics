import os
import inspect
import numpy as np
import matplotlib.pyplot as plt

import uuid 
import pickle as pkl

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/gym_thing/gym_thing/")
from pusher_env import ThingPusher
from visual_pusher_env import ThingVisualPusher
import pybullet as p

def write_file_pkl(data, name, location="."):
    tag = uuid.uuid4().hex[:].upper()
    filename = os.path.join(location, name + "_" + tag)
    os.makedirs(location, exist_ok=True)
    with open(f'{filename}.pkl', 'wb') as f:
        pkl.dump(data, f)

def magnitude_experiment():
    n_steps = 32
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False, substeps=n_steps)
    
    s_list = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    n = len(s_list)
    ll = 30
      
    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
    }
    
    for ii, s in enumerate(s_list):
        env.reset()
        for jj in range(ll): 
            obs, reward, done, info = env.step(action=np.array([0.50 * s, 0]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]

    return data

def horizontal_position_experiment():
    n_steps = 32
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False, substeps=n_steps)
    
    s_list = list(3 * np.arange(-0.050, 0.060, 0.01))
    n = len(s_list)
    ll = 30

    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
    }
    
    for ii, s in enumerate(s_list):
        env.reset()
        for _ in range(5):
            obs, reward, done, info = env.step(action=np.array([0, s]))

        for jj in range(ll): 
            obs, reward, done, info = env.step(action=np.array([0.50, 0]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]
    return data

def weight_experiment():   
    n_steps = 32 
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False, substeps=n_steps)
    # s_list = [5, 6, 7, 8, 9, 10]   
    s_list = [5]
    n = len(s_list)
    ll = 20

    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
    }
    
    for ii, s in enumerate(s_list):
        env.reset()
        env._p.changeDynamics(env.scene.plateId, -1, mass=s)
        for jj in range(ll): 
            obs, reward, done, info = env.step(action=np.array([0.50, 0]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]

    return data

def acceleration_experiment(frame_skip):
    n = 10
    n_steps = 32
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False, substeps=n_steps, frame_skip=frame_skip)
    ll = 30
    p = np.random.uniform(-1,1,2) 
        
    data = {
        "img": np.zeros((n, ll, 64, 64, 3), dtype=np.uint8), 
        "ft": np.zeros((n, ll, n_steps, 6)), 
        "arm": np.zeros((n, ll, n_steps, 6)),
    }
    
    for ii in range(n):
        env.reset()
        for _ in range(int(30 / frame_skip)):
                obs, reward, done, info = env.step(action=np.array([0.40, 0]))
        for jj in range(ll): 
            p = np.random.uniform(-1,1,2)
            obs, reward, done, info = env.step(action=np.array([p[0], p[1]]))
            data["img"][ii, jj] = obs["img"]
            data["ft"][ii, jj] = obs["ft"]
            data["arm"][ii, jj] = obs["arm"]

    return data

if __name__ == "__main__":
    # data = magnitude_experiment()
    # data = horizontal_position_experiment()
    # data = weight_experiment()
    data = acceleration_experiment(frame_skip=2)

    write_file_pkl(data=data, name="acceleration_experiment_2frameskip_32substeps", location="./data/ft_sim/")