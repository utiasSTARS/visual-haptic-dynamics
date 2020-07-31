"""
Test environment with simple inputs.
"""
import os, sys, inspect, time
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/")

from classic_control_pixel import VisualPendulum
from gym_thing.gym_thing import reacher_env, pusher_env, visual_pusher_env, visual_reacher_env
import gym
import numpy as np

def test_actions_pendulum():
    env = VisualPendulum(frame_stack=0)
    env.reset(np.array([np.pi,0]))

    for ii in range(10000000):
        obs, _, _, _ = env.step(np.array([1.0]))
        if ii % 100 == 0:
            env.reset()
            print("resetting")
        # print("Obs received: ", obs.shape)
        # print("Action dim: ", env.action_space.shape)
        # print("Obs dim: ", env.observation_space.shape)
        # print("Reset img output: ", img.shape)

    env.close()

def test_gym_reacher():
    env = gym.make('ThingReacher2D-v0', is_render=False)
    for ii in range(10000000):
        p = np.random.uniform(-1,1,2)
        obs, _, _, _ = env.step(np.array([p[0], p[1]]))
        if ii % 150 == 0:
            tic = time.time()
            env.reset()
            reset_time = time.time() - tic
            print(f"Resetting with {ii} time steps, time {reset_time}")
        # print("Obs received: ", obs.shape)
        # print("Action dim: ", env.action_space)
        # print("Obs dim: ", env.observation_space)

def test_gym_pusher():
    env = gym.make('ThingPusher-v0', is_render=False)
    for ii in range(10000000):
        p = np.random.uniform(-1,1,2)
        obs, _, _, _ = env.step(np.array([p[0], p[1]]))
        if ii % 150 == 0:
            tic = time.time()
            env.reset()
            reset_time = time.time() - tic
            print(f"Resetting with {ii} time steps, time {reset_time}")
        print("Obs received: ", obs)
        # print("Action dim: ", env.action_space)
        # print("Obs dim: ", env.observation_space)

def test_gym_visual_reacher():
    env = gym.make('ThingVisualReacher2D-v0', is_render=False)
    for ii in range(10000000):
        p = np.random.uniform(-1,1,2)
        obs, _, _, _ = env.step(np.array([p[0], p[1]]))
        if ii % 150 == 0:
            tic = time.time()
            env.reset()
            reset_time = time.time() - tic
            print(f"Resetting with {ii} time steps, time {reset_time}")
        # print("Action dim: ", env.action_space)
        # print("Obs dim: ", env.observation_space)

def test_gym_visual_pusher():
    env = gym.make('ThingVisualPusher-v0', is_render=False)
    for ii in range(10000000):
        p = np.random.uniform(-1,1,2)
        obs, _, _, _ = env.step(np.array([p[0], p[1]]))
        if (ii + 1) % 150 == 0:
            tic = time.time()
            env.reset()
            reset_time = time.time() - tic
            print(f"Resetting with {ii} time steps, time {reset_time}")
        # print("Action dim: ", env.action_space)
        # print("Obs dim: ", env.observation_space)

if __name__=="__main__":
    # test_gym_reacher()
    test_gym_pusher()
    # test_gym_visual_reacher()
    # test_gym_visual_pusher()