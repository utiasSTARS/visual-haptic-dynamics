import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/gym_thing/gym_thing/")
from pusher_env import ThingPusher
from visual_pusher_env import ThingVisualPusher

def collect_ft_data():
    env = ThingVisualPusher(render_w=64, render_h=64, goal_vis=False)
    env.reset()
    for ii in range(100000): 
        if ii % 100 == 0: 
            print(f"Time step {ii}")
        # if ii < 5:
            # obs, reward, done, info = env.step(action=np.array([0, -0.025]))

        obs, reward, done, info = env.step(action=np.array([0.10, 0]))
        # print(obs["ft"][:, 0:3])

if __name__ == "__main__":
    collect_ft_data()