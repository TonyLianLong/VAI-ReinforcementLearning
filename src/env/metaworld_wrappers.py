import numpy as np
from numpy.random import randint
import os
import gym
import matplotlib.pyplot as plt
import random
import mujoco_py
import metaworld

import cv2
from collections import deque

from scipy.ndimage.filters import gaussian_filter

def make_pad_env(
        domain_name,
        task_name='drawer-close-v1',
        seed=0,
        episode_length=1000, # Not supported, length is set by the env
        frame_stack=3,
        action_repeat=4,
        mode='train',
        action_factor=1.0, # Not supported
        action_bias=0, # Not supported
        action_noise_factor=0, # Not supported
        moving_average_denoise=False,
        moving_average_denoise_factor=None,
        moving_average_denoise_alpha=None,
        exponential_moving_average=0.0
    ):
    """Make environment for PAD experiments"""
    assert domain_name == "metaworld"
    if mode == "train":
        mode = "grid"
    # All environments:
    # print(metaworld.MT1.ENV_NAMES)
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](texture=mode)
    env = MetaWorldWrapper(env, mt1, seed, action_repeat)
    env = FrameStack(env, frame_stack)
    env = MovingAverageWrapper(env, mode, moving_average_denoise=moving_average_denoise, moving_average_denoise_factor=moving_average_denoise_factor, moving_average_denoise_alpha=moving_average_denoise_alpha, k=frame_stack)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    assert exponential_moving_average == 0, "exponential_moving_average is not supported in DrawerWorld"

    return env


class MovingAverageWrapper(gym.Wrapper):
    """Use moving average to de-noise"""
    def __init__(self, env, mode, moving_average_denoise, moving_average_denoise_factor, moving_average_denoise_alpha, k, reset_avg=False):
        # k needs to be set to frame_stack, reset_avg should be set to False for tasks with targets

        gym.Wrapper.__init__(self, env)
        self._mode = mode

        if moving_average_denoise:
            self.factor = int(moving_average_denoise_factor * 255)
            obs_shape = env.observation_space.shape
            self.avg_shape = (obs_shape[0] // k,) + obs_shape[1:]
            self._reset_avg()
            self.k = k
            self.alpha = moving_average_denoise_alpha
            self.reset_avg = reset_avg

        self.enabled = moving_average_denoise
        
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        if not self.enabled:
            return obs
        if self.reset_avg: # reset avg across resets
            self._reset_avg()
        self._update_avg(obs)
        return obs # No de-noise

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # import ipdb; ipdb.set_trace()
        if not self.enabled:
            return obs, reward, done, info
        info["un_denoised_obs"] = obs
        ret = self._get_obs(obs), reward, done, info
        self._update_avg(obs) # update obs after get to avoid removing the foreground
        return ret

    def _reset_avg(self):
        self.sum = np.zeros(self.avg_shape)
        self.count = 0

    def _update_avg(self, obs):
        mean_obs = obs.reshape((-1, ) + self.avg_shape).mean(axis=0)
        self.sum += mean_obs
        self.count += 1

    def _get_obs(self, obs):
        # import ipdb; ipdb.set_trace()
        aggressiveness = self.alpha
        avg = self.sum / self.count
        # avg = gaussian_filter(avg, sigma=3)
        mean_color = avg.reshape((self.avg_shape[0], -1)).mean(axis=1).reshape((self.avg_shape[0], 1, 1))
        de_mean_obs = obs - np.tile(avg, (self.k, 1, 1)) * aggressiveness
        de_mean_obs = de_mean_obs * (de_mean_obs > self.factor)
        ret = de_mean_obs + np.tile(mean_color, (self.k, 1, 1)) * aggressiveness
        return np.clip(ret, a_min=0.0, a_max=255.0).astype('uint8')

class MetaWorldWrapper(gym.Wrapper):
    """1. deal with action_repeat (may smooth out actions as todo) 2. render"""
    def __init__(self, env, mt1, seed, action_repeat, action_momentum=0.6, smooth=False):
        gym.Wrapper.__init__(self, env)
        device_id = int(os.environ.get("EGL_DEVICE_ID", 0))
        print("Rendering on", device_id)
        """env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, -1, device_id=device_id) 
        env.viewer.cam.azimuth = 205
        env.viewer.cam.elevation = -170
        env.viewer.cam.distance = 2.3
        env.viewer.cam.lookat[0] = 1.05
        env.viewer.cam.lookat[1] = 1.15
        env.viewer.cam.lookat[2] = -0.1"""
        env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, -1, device_id=device_id) 
        env.viewer.cam.azimuth = 180
        env.viewer.cam.elevation = -90
        env.viewer.cam.distance = 0.96
        env.viewer.cam.lookat[0] = 0
        env.viewer.cam.lookat[1] = 0.6
        env.viewer.cam.lookat[2] = 0

        random.seed(seed)
        env.seed(seed)
        w, h = 100, 100
        self.train_tasks = mt1.train_tasks
        self.smooth = smooth
        if self.smooth: # Action smoothing
            self.action_momentum = action_momentum
            self.last_action = None
        
        self.action_repeat = action_repeat
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((3, w, h)),
            dtype=np.uint8
        )
        self._max_episode_steps = env.max_path_length
        self.step_count = 0

    def reset(self):
        task = random.choice(self.train_tasks)
        #task = self.train_tasks[0]

        self.env.set_task(task) 
        self.env.reset()
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        reward_sum = 0
        if self.smooth:
            if self.last_action is None:
                self.last_action = action
            else:
                action = self.last_action * self.action_momentum + action * (1 - self.action_momentum)
                self.last_action = action
        for _ in range(self.action_repeat):
            _, reward, done, info = self.env.step(action)
            reward_sum += reward
            self.step_count += 1
            if self.step_count == self._max_episode_steps:
                done = True
            if done:
                break
        return self._get_obs(), reward_sum, done, info

    def _get_obs(self):
        w, h = 100, 100 # no self
        self.env.viewer.render(width=w, height=h)
        obs = np.transpose(self.env.viewer.read_pixels(width=w, height=h, depth=False), (2, 0, 1))
        return np.ascontiguousarray(obs)

class FrameStack(gym.Wrapper):
    """Stack frames as observation"""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
