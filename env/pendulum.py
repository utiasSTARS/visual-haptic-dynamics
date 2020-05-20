import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

class VisualPendulum(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, render_w=64, render_h=64, g=10.0, m=1., l=1.):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = m
        self.l = l
        self.viewer = None

        self.render_w = render_w
        self.render_h = render_h

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, 
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(render_w, render_h, 1), 
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th) ** 2 \
            + .1 * thdot ** 2 \
            + .001 * (u ** 2)

        newthdot = thdot \
            + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi)
            + 3. / (self.m * self.l ** 2) * u) * self.dt
        newth = th + newthdot * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}
    
    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

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

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)