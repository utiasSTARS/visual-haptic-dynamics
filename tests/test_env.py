"""
Test environment with simple inputs.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../env/')))

from pendulum import VisualPendulum
import numpy as np

def test_actions():
    env = VisualPendulum(frame_stack=0)
    env.reset(np.array([np.pi,0]))

    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0]))
        img = env.reset()
        print("Obs received: ", obs.shape)
        print("Action dim: ", env.action_space.shape)
        print("Obs dim: ", env.observation_space.shape)
        print("Reset img output: ", img.shape)

    env.close()

if __name__=="__main__":
    test_actions()