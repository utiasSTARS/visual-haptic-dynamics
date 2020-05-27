"""
Test latent environment with roll out.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../env/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pendulum import LatentPendulum
import numpy as np

from utils import load_models, frame_stack
from argparse import Namespace
import json
from torchvision import transforms
from utils import (set_seed_torch, Normalize)
set_seed_torch(3)

def test_actions():
    # Load model weights
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models/'))
    models = {}
    for filedir in os.listdir(models_dir):
        fullpath = os.path.join(models_dir, filedir)
        if os.path.isdir(fullpath):
            models[fullpath] = {}
            with open(os.path.join(fullpath, 'hyperparameters.txt'), 'r') as fp:
                models[fullpath]['hyperparameters'] = Namespace(**json.load(fp))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        Normalize(mean=0.27, var=1.0 - 0.27) # 64x64
        ])

    for path, model in models.items():
        model_args = model['hyperparameters']
        trained_models = load_models(path, model_args, mode='eval', device='cpu')
        env = LatentPendulum(models=trained_models, device='cpu', img_transform=transform)
    # env.reset(np.array([np.pi,0]))

    # for _ in range(1):
    #     obs, _, _, _ = env.step(np.array([0]))
    #     img = env.reset()
    #     print("Obs received: ", obs.shape)
    #     print("Action dim: ", env.action_space.shape)
    #     print("Obs dim: ", env.observation_space.shape)
    #     print("Reset img output: ", img.shape)

    # env.close()

if __name__=="__main__":
    test_actions()