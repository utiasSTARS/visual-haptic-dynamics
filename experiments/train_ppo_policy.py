import os, sys, inspect, time
sys.path.append('..')
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/")
from gym_thing.gym_thing import reacher_env, pusher_env, visual_pusher_env, visual_reacher_env
from utils import set_seed_torch, rgb2gray, FrameStack
from args.parser import parse_ppo_args
import gym
from ppo import PPO
from models import ActorCriticMLP, ActorCriticCNN
import torch 
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train(args):
    # creating environment
    if args.is_render is not None:
        env = gym.make(args.env_name, is_render=args.is_render, render_h=args.dim_x[0], render_w=args.dim_x[1])
    else:
        env = gym.make(args.env_name)

    if args.architecture =="cnn":
        env = FrameStack(env, k=args.frame_stack)

    state_dim = env.observation_space.shape[0] # 24
    action_dim = env.action_space.shape[0] # 4
    print("State dim: {}".format(env.observation_space.shape))
    print("Action dim: {}".format(env.action_space.shape))

    if args.random_seed:
        print("Random Seed: {}".format(args.random_seed))
        env.seed(args.random_seed)
        set_seed_torch(args.random_seed)  

    memory = Memory()

    if args.architecture == "mlp":
        actor_critic = ActorCriticMLP(
            state_dim, 
            action_dim, 
            args.action_std
            ).to(args.device)
    elif args.architecture =="cnn":
        actor_critic = ActorCriticCNN(
            state_dim, 
            action_dim, 
            args.action_std, 
            img_dim=(args.dim_x[0], args.dim_x[1], args.frame_stack * args.dim_x[2])
            ).to(args.device)

    ppo = PPO(
        args.lr, 
        args.gamma, 
        args.epochs, 
        args.eps_clip, 
        args.device,
        actor_critic=actor_critic
    )
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    avg_time = 0

    # training loop
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()

        tic = time.time()
        for t in range(args.max_timesteps):
            time_step +=1

            # Running policy_old:            
            if args.architecture =="cnn":
                if args.dim_x[-1] == 1:
                    state = rgb2gray(state)

                state = state.transpose(0, 3, 1, 2)
                state = state.reshape(-1, *state.shape[2:])[np.newaxis]
                action = ppo.select_action(state, memory)
            else:
                action = ppo.select_action(state[np.newaxis], memory)

            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % args.update_timestep == 0:
                print("Updating policy")
                ppo.update(memory)
                print("Policy updated!")
                memory.clear_memory()
                print("Memory cleared")
                time_step = 0
            running_reward += reward
            if args.render:
                env.render()
            if done:
                break
        episode_t = time.time() - tic

        avg_length += t
        avg_time += episode_t

        if i_episode % args.logging_interval == 0:
            avg_length = int(avg_length/args.logging_interval)
            running_reward = int((running_reward/args.logging_interval))
            avg_time = float(avg_time/args.logging_interval)
            
            print('Episode {} \t Avg length: {} \t Avg reward: {} \t Avg runtime per episode: {}'\
                    .format(i_episode, avg_length, running_reward, avg_time))

            if running_reward > args.solved_reward:
                print("########## Solved! ##########")
                break

            running_reward = 0
            avg_length = 0
            avg_time = 0

def main():
    args = parse_ppo_args()
    train(args)

if __name__ == '__main__':
    main()
    