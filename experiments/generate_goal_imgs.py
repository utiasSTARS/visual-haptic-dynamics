import os, sys
os.sys.path.insert(0, "..")

import torch
from utils import frame_stack
from datasets import VisualHaptic
import matplotlib.pyplot as plt
import pickle

def generate_goal_raw_data(dataset, idx, context="joint", device="cpu"):
    data = dataset[idx] # (16, 1, 64, 64)
    
    img = torch.from_numpy(data["img"]).unsqueeze(0)
    haptic = torch.from_numpy(data["ft"]).unsqueeze(0)
    arm = torch.from_numpy(data["arm"]).unsqueeze(0)
    gt_plate_pos = torch.from_numpy(data["gt_plate_pos"])

    if context == "joint": 
        context_data = torch.cat((haptic, arm), dim=-1) # (n, l, f, 12)
    elif context == "ft":
        context_data = haptic
    elif context == "arm":
        context_data = arm

    context_data = context_data.float().to(device=device) # (n, l, f, 6)
    context_data = context_data.transpose(-1, -2)
    context_data = context_data[:, 1:]

    gt_plate_pos = gt_plate_pos[1:]

    img = img.float().to(device=device)
    img = frame_stack(img, frames=1)

    return img[:, -1], context_data[:, -1], gt_plate_pos[-1, :]

if __name__ == "__main__":
    datapath="/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_withGT_3D9E4376CF4746EEA20DCD520218038D.pkl"
    dataset = VisualHaptic(datapath)
    goals = []

    for ii in range(len(dataset)):
        goal = generate_goal_raw_data(dataset, ii)
        goals.append(goal)

    with open('./goal_imgs/goals.pkl', 'wb') as handle:
        pickle.dump(goals, handle)