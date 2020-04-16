import cv2
import numpy as np

import gym
from gym import spaces
from baselines.common.vec_env import VecEnvWrapper



class ProcgenWrapper(VecEnvWrapper):
    def __init__(self, venv, wrapframe=True, width=84, height=84, num_colors=1, op=[2, 0, 1]):
        super().__init__(venv,
                        observation_space=venv.observation_space["rgb"],
                        action_space=venv.action_space)
        self.wrapframe = wrapframe
        if wrapframe:
            self.width = width
            self.height = height
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, num_colors),
                dtype=np.uint8,
            )


        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            1.0, [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ])
            #dtype=self.observation_space.dtype)
        self.high = int(self.venv.observation_space["rgb"].high.max())
    
    def step_async(self, actions):
        # print("step")
        return self.venv.step_async(np.squeeze(actions, axis=1))

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = obs["rgb"]
        if self.wrapframe:
            obs = self.wrapframe(obs)
        return self.transpose(obs), rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        obs = obs["rgb"]
        if self.wrapframe:
            obs = self.wrapframe(obs)
        return self.transpose(obs)
    
    def wrapframe(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        obs = np.expand_dims(frame, -1)
        return obs

    def transpose(self, obs):
        # vf = np.vectorize(lambda x: x.transpose(self.op[0], self.op[1], self.op[2])/self.high)
        return obs.transpose(0, self.op[0]+1, self.op[1]+1, self.op[2]+1)/self.high
