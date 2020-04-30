"""
Test environment with simple inputs.
"""
import os, sys, time
sys.path.append('../env/')

from pendulum import Pendulum
import numpy as np

def test_actions():
  env = Pendulum()
  env.reset(np.array([np.pi,0]))

  while True:
    for _ in range(100):
      obs, _, _, _ = env.step(np.array([0]))
    env.reset()

if __name__=="__main__":
  test_actions()