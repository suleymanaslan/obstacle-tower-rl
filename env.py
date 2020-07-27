# adapted from https://github.com/Kaixhin/Rainbow

from collections import deque
import random
import torch
import cv2
import gym
import numpy as np
from gym.wrappers.pixel_observation import PixelObservationWrapper


class Env():
    def __init__(self, action_size, history_length, pixel_obs=True):
        self.device = torch.device("cuda:0")
        self.wrapped_env = PixelObservationWrapper(gym.make("LunarLander-v2"), pixels_only=True) if pixel_obs else gym.make("LunarLander-v2")
        self.action_space = [i for i in range(action_size)]
        self.window = history_length
        self.state_buffer = deque([], maxlen=self.window)
        self.pixel_obs = pixel_obs
    
    def _reset_buffer(self):
        for _ in range(self.window):
            if self.pixel_obs:
                self.state_buffer.append(torch.zeros(84, 84, device=self.device))
            else:
                self.state_buffer.append(torch.zeros(8, device=self.device))
    
    def _process_observation(self, observation):
        if self.pixel_obs:
            observation = cv2.cvtColor(cv2.resize(observation["pixels"], (84, 84), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255)
        else:
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return observation
    
    def reset(self):
        self._reset_buffer()
        observation = self.wrapped_env.reset()
        observation = self._process_observation(observation)
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)
    
    def close(self):
        self.wrapped_env.close()
    
    def _step(self, action, frame_buffer):
        reward = 0
        for t in range(4):
            observation_t, reward_t, done, info = self.wrapped_env.step(action)
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
        if self.pixel_obs:
            frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        else:
            frame_buffer = torch.zeros(2, 8, device=self.device)
        return self._step(action, frame_buffer)
