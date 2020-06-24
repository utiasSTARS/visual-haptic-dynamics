"""
Test environment with simple inputs.
"""
import os, sys, time, inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/")

from classic_control_pixel import VisualPendulum
from gym_thing.gym_thing import reacher_env
import gym
import numpy as np

def test_actions_pendulum():
    env = VisualPendulum(frame_stack=0)
    env.reset(np.array([np.pi,0]))

    for ii in range(1000):
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
    env = gym.make('ThingReacher2D-v0')
    for ii in range(1000):
        print(ii)
        obs, _, _, _ = env.step(np.array([1.0]))
        if ii % 100 == 0:
            env.reset()
            print("resetting")
        print("Obs received: ", obs.shape)
        print("Action dim: ", env.action_space)
        print("Obs dim: ", env.observation_space)


def test_gym_pusher():
    env = gym.make('ThingPusher-v0')
    for ii in range(1000):
        print(ii)
        obs, _, _, _ = env.step(np.array([0.1]))
        if ii % 100 == 0:
            env.reset()
            print("resetting")
        print("Obs received: ", obs.shape)
        print("Action dim: ", env.action_space)
        print("Obs dim: ", env.observation_space)

def test_gym_visual_reacher():
    env = gym.make('ThingVisualReacher2D-v0')
    for ii in range(1000):
        print(ii)
        obs, _, _, _ = env.step(np.array([0.1]))
        if ii % 100 == 0:
            env.reset()
            print("resetting")
        print("Action dim: ", env.action_space)
        print("Obs dim: ", env.observation_space)

def test_gym_visual_pusher():
    env = gym.make('ThingVisualPusher-v0')
    for ii in range(1000):
        print(ii)
        obs, _, _, _ = env.step(np.array([0.25, 0]))
        if ii % 100 == 0:
            env.reset()
            print("resetting")
        print("Action dim: ", env.action_space)
        print("Obs dim: ", env.observation_space)

if __name__=="__main__":
    # test_actions()
    # test_gym_reacher()
    # test_gym_pusher()
    # test_gym_visual_reacher()
    test_gym_visual_pusher()