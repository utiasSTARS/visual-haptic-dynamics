"""
This script generates trajectories using agent trained with SAC-X.
It replaces the trained Q-scheduler with the U-scheduler for exploration data.
"""

import _pickle as pickle
import argparse
import gym
import math
import numpy as np
import os

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.algorithms.sac_x.schedulers_update.q_scheduler import QTableScheduler
from rl_sandbox.envs.utils import make_env
from rl_sandbox.model_architectures.utils import make_model
from rl_sandbox.utils import set_seed


class UScheduler:
    def __init__(self,
                 num_tasks):
        self._num_tasks = num_tasks

    def compute_action(self, state, h):
        return np.random.randint(self._num_tasks, size=(1,)), None, [np.nan], None, None, None, None


def generate_trajectories(agent, env, preprocess, num_trajectories, trajectory_length, save_path, render_h, render_w, n_steps):
    data = {
        "img": np.zeros((num_trajectories, trajectory_length + 1, render_h, render_w, 3), dtype=np.uint8),
        "ft": np.zeros((num_trajectories, trajectory_length + 1, n_steps, 6)),
        "arm": np.zeros((num_trajectories, trajectory_length + 1, n_steps, 6)),
        "action": np.zeros((num_trajectories, trajectory_length, 2)),
        "reward": np.zeros((num_trajectories, trajectory_length)),
        "gt_plate_pos": np.zeros((num_trajectories, trajectory_length, 3))
    }

    for traj_i in range(num_trajectories):
        env.reset()
        h_state = agent.reset()
        obs, reward, done, info = env.step(action=np.random.randn(2))
        data["img"][traj_i, 0] = info["infos"][-1]['original_obs']['img']
        data["ft"][traj_i, 0] = info["infos"][-1]['original_obs']['ft']
        data["arm"][traj_i, 0] = info["infos"][-1]['original_obs']['arm']

        for obs_i in range(trajectory_length):
            obs = preprocess(obs)
            action, h_state, act_info = agent.compute_action(obs=obs,
                                                             hidden_state=h_state)
            action = np.clip(action, a_min=-1, a_max=1)
            obs, reward, done, info = env.step(action)
            data["img"][traj_i, obs_i + 1] = info["infos"][-1]['original_obs']['img']
            data["ft"][traj_i, obs_i + 1] = info["infos"][-1]['original_obs']['ft']
            data["arm"][traj_i, obs_i + 1] = info["infos"][-1]['original_obs']['arm']
            data["action"][traj_i, obs_i] = action
            data["reward"][traj_i, obs_i] = reward
            data["gt_plate_pos"][traj_i, obs_i] = info["infos"][-1]["achieved_goal"]

    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def main(args):
    assert args.num_trajectories > 0
    assert os.path.isfile(args.model_path)
    assert os.path.isfile(args.config_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    set_seed(args.seed)
    with open(args.config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    env_setting[c.ENV_BASE]["substeps"] = args.n_steps
    env = make_env(env_setting, seed=args.seed)
    intentions = make_model(config[c.INTENTIONS_SETTING])

    scheduler = UScheduler(num_tasks=config[c.NUM_TASKS])

    agent = SACXAgent(scheduler=scheduler,
                      intentions=intentions,
                      learning_algorithm=None,
                      scheduler_period=args.scheduler_period,
                      preprocess=config[c.EVALUATION_PREPROCESSING])

    generate_trajectories(agent,
                          env,
                          config[c.BUFFER_PREPROCESSING],
                          args.num_trajectories,
                          args.trajectory_length,
                          args.save_path,
                          env_setting[c.ENV_BASE][c.RENDER_H],
                          env_setting[c.ENV_BASE][c.RENDER_W],
                          args.n_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the trajectories")
    parser.add_argument("--num_trajectories", required=True, type=int, help="The number of trajectories")
    parser.add_argument("--n_steps", required=True, type=int, help="The number of substeps")

    parser.add_argument("--trajectory_length", required=True, type=int, help="The trajectory length")
    parser.add_argument("--scheduler_period", required=True, type=int, help="The interval in which the next intention is chosen")
    args = parser.parse_args()

    main(args)
