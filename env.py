# adapted from https://github.com/Kaixhin/Rainbow

from collections import deque
import time
import random
import torch
import cv2
import gym
import numpy as np
from gym.wrappers.pixel_observation import PixelObservationWrapper
from obstacle_tower_env import ObstacleTowerEnv as ObstacleTower
from IPython import display
from PIL import Image


class Env():
    def __init__(self, action_size, history_length):
        self.device = torch.device("cuda:0")
        self.wrapped_env = self._get_env()
        self.action_space = [i for i in range(action_size)]
        self.window = history_length
        self.state_buffer = deque([], maxlen=self.window)
    
    def _get_env(self):
        return PixelObservationWrapper(gym.make("LunarLander-v2"), pixels_only=True)
    
    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))
    
    def _process_observation(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation["pixels"], (84, 84), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255)
        return observation
    
    def render(self):
        self.wrapped_env.render()
    
    def reset(self):
        self._reset_buffer()
        observation = self.wrapped_env.reset()
        observation = self._process_observation(observation)
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)
    
    def close(self):
        self.wrapped_env.close()
    
    def _step(self, action, frame_buffer, render=False):
        reward = 0
        for t in range(4):
            observation_t, reward_t, done, info = self.wrapped_env.step(action)
            if render:
                self.render()
            reward += reward_t
            if t == 2:
                frame_buffer[0] = self._process_observation(observation_t)
            elif t == 3:
                frame_buffer[1] = self._process_observation(observation_t)
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0), reward, done, info
    
    def step(self, action):
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        return self._step(action, frame_buffer)


class SimpleEnv(Env):
    def __init__(self, action_size, history_length):
        super(SimpleEnv, self).__init__(action_size, history_length)
    
    def _get_env(self):
        return gym.make("LunarLander-v2")
    
    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(8, device=self.device))
    
    def _process_observation(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return observation
    
    def step(self, action):
        frame_buffer = torch.zeros(2, 8, device=self.device)
        return self._step(action, frame_buffer)


class ObstacleTowerEnv(Env):
    def __init__(self, action_size, history_length):
        super(ObstacleTowerEnv, self).__init__(action_size, history_length)
        self.movement_dict = {0:"No-Op", 1:"Forward", 2:"Backward"}
        self.cam_rot_dict = {0:"No-Op", 1:"Counter-Clockwise", 2:"Clockwise"}
        self.jump_dict = {0:"No-Op", 1:"Jump"}
        self.turn_dict = {0:"No-Op", 1:"Right", 2:"Left"}
    
    def _get_env(self):
        return ObstacleTower(retro=True, realtime_mode=False, greyscale=True)
    
    def seed(self, seed):
        self.wrapped_env.seed(seed)
    
    def floor(self, floor):
        self.wrapped_env.floor(floor)
    
    def action_to_mda(self, action, simple_action=False):
        if simple_action:
            movement = action // 3
            cam_rot = action % 3
            jump = 0
            turn = 0
        else:
            movement = action // 18
            cam_rot = (action // 6) % 3
            jump = (action // 3) % 2
            turn = action % 3
        return np.array([movement, cam_rot, jump, turn])
    
    def mda_to_discrete(self, mda):
        return mda[0] * 18 + mda[1] * 6 + mda[2] * 3 + mda[3]
    
    def render(self):
        render_img = cv2.resize(self.wrapped_env.render(), (0,0), fx=4.0, fy=4.0)
        display.clear_output(wait=True)
        display.display(Image.fromarray(render_img))
        time.sleep(1/60)
    
    def step(self, action, simple_action=False, render=False):
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        mda_action = self.action_to_mda(action, simple_action)
        return self._step(self.mda_to_discrete(mda_action), frame_buffer, render)
    
    def _process_observation(self, observation):
        observation = observation.squeeze()
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255)
        return observation
