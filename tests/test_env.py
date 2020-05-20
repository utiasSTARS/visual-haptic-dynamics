"""
Test environment with simple inputs.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../env/')))

from pendulum import VisualPendulum
import numpy as np

def test_actions():
    env = VisualPendulum()
    env.reset(np.array([np.pi,0]))

    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0]))
        env.reset()
        print(obs.shape)

if __name__=="__main__":
    test_actions()