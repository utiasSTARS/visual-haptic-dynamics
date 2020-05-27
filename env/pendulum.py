import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from collections import deque 

from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

import torch

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class VisualPendulum(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, render_w=64, render_h=64, g=10.0, m=1., l=1., frame_stack=0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = m
        self.l = l
        self.viewer = None

        self.render_w = render_w
        self.render_h = render_h
        self.frame_stack = frame_stack
        self.img_buffer = deque([], maxlen=1 + self.frame_stack)

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, 
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(1 + frame_stack, render_w, render_h, 3), 
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cost(self, th, thdot, u):
        return angle_normalize(th) ** 2 \
            + .1 * thdot ** 2 \
            + .001 * (u ** 2) 

    def _step(self, u):
        th, thdot = self.state # th := theta

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = self.cost(th=th, thdot=thdot, u=u)

        newthdot = thdot \
            + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi)
            + 3. / (self.m * self.l ** 2) * u) * self.dt
        newth = th + newthdot * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        
        return self._get_obs(), -costs

    def step(self, u):
        new_img, costs = self._step(u)
        self.img_buffer.appendleft(new_img)
        img = np.stack(list(self.img_buffer))
        return img, costs, False, {}
    
    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None

        self.img_buffer.appendleft(self._get_obs())

        # Let system evolve until we have enough history
        for _ in range(self.frame_stack):
            self.img_buffer.appendleft(self._step(np.array([0]))[0])

        img = np.stack(list(self.img_buffer))
        return img

    def _image_transforms(self, img):
        resized_img = resize(img, (self.render_w, self.render_h), anti_aliasing=True)
        rescaled_img = 255 * resized_img
        img = rescaled_img.astype(np.uint8) 
        return img

    def _get_obs(self):
        img = self.render(mode='rgb_array') 
        img = self._image_transforms(img)
        return img

    def render(self, mode='human'):
        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            self.imgtrans = rendering.Transform()

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class LatentPendulum(VisualPendulum):
    """Wrap environment with latent dynamics model and reward function of choice."""
    def __init__(self, models, device, frame_stack=1, img_transform=None):
        super().__init__(frame_stack=frame_stack)
        self.enc, self.dec, self.dyn = models
        self.device = device
        self.img_transform = img_transform

        # Encode goal image
        goal_img = self.reset(np.array([0., 0.]))
        with torch.no_grad():
            self.z_goal, _, _ = self.encode(goal_img)
        
    def to(self, device):
        self.device = device
        self.enc.to(device=device).eval()
        self.dec.to(device=device).eval()
        self.dyn.to(device=device).eval()

    def encode(self, img):
        """Encode image of dimension: (frame_stack, w, h, c)"""
        transformed_img = torch.zeros((1, 2, 64, 64))
        for ii in range(img.shape[0]):
            out = self.img_transform(img[ii])
            transformed_img[:, ii, :, :] = out
        
        return self.enc(transformed_img)

    def rollout(self, actions, loss="euclidean"):
        H = actions.shape[0] # (H, dim_u)
        img_t = np.stack(list(self.img_buffer))        
        z_t, mu_t, logvar_t = self.encode(img_t) # (1, dim_z)
        var_t = torch.diag_embed(torch.exp(logvar_t)) # (1, dim_z, dim_z)

        z = torch.zeros((H + 1, z_t.shape[-1]))
        z[0] = z_t

        for ii in range(H):
            z_t1, mu_t1, var_t1, _ = self.dyn(
                z_t=z_t, mu_t=mu_t, var_t=var_t, 
                u=actions[ii].unsqueeze(0), single=True
            )
            z[ii + 1] = z_t1

        if loss == "euclidean":
            cost = self.euclidean_cost(z)
        elif loss == "metric":
            cost = self.manifold_curve_energy(z)
        else:
            raise NotImplementedError()

        return cost

    def euclidean_cost(self, z):
        cost = (self.z_goal - z)**2
        return cost.sum()

    def manifold_curve_energy(self, z):
        z_all = torch.cat((z, self.z_goal))
        T = z.shape[0]
        dt = 1. / T
        energy = 0.5 * torch.sum(dt * torch.sum((z_all[1:] - z_all[:-1])**2, dim=-1))
        return energy