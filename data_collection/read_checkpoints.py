import _pickle as pickle
import argparse
import cv2
import gzip
import matplotlib.pyplot as plt
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True, type=str, help="The directory containing the checkpoint files")
parser.add_argument("--convert", action="store_true", help="Convert the checkpoint files such that each pickle file is a single trajectory")
parser.add_argument("--visualize", action="store_true", help="Visualize the trajectories")
parser.add_argument("--save_dir", type=str, help="The directory storing the converted pickle files")
args = parser.parse_args()

assert os.path.isdir(args.checkpoint_path)
assert not args.convert or (args.convert and args.save_dir)


if args.save_dir and not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

num_episodes = 0
curr_episode_obss = []
curr_episode_acts = []
curr_episode_rews = []

for checkpoint in os.listdir(args.checkpoint_path):
    with gzip.open(os.path.join(args.checkpoint_path, checkpoint), "rb") as chkpt_file:
        chkpt_data = pickle.load(chkpt_file)

    for idx, obs in enumerate(chkpt_data['observations']):
        curr_episode_obss.append(obs)
        curr_episode_acts.append(chkpt_data['actions'][idx])
        curr_episode_rews.append(chkpt_data['rewards'][idx])

        if args.visualize:
            cv2.namedWindow('ThingVisualPusher Images - Episode: {}'.format(num_episodes), cv2.WINDOW_NORMAL)
            images = []
            time.sleep(0.05)
            cv2.imshow('ThingVisualPusher Images - Episode: {}'.format(num_episodes), obs[:-12].reshape((84, 84)))
            cv2.waitKey(1)

        if chkpt_data['dones'][idx]:
            if args.convert:
                with gzip.open(os.path.join(args.save_dir, "{}.pkl".format(num_episodes)), "wb") as ep_file:
                    pickle.dump({
                        "observations": curr_episode_obss,
                        "actions": curr_episode_acts,
                        "rewards": curr_episode_rews,
                    }, ep_file)

            num_episodes += 1
            curr_episode_obss = []
            curr_episode_acts = []
            curr_episode_rews = []
            cv2.destroyAllWindows()

            if num_episodes % 100 == 0:
                print("Processed {} episodes".format(num_episodes))
