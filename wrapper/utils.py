import gym
from gym import spaces
import gvgai
from baselines.common.wrappers import TimeLimit

def make_gvgai(env_id, max_episode_steps=None):
    env = gym.make(env_id, pixel_observations=True, tile_observations=False, include_semantic_data=True)
    env = TimeLimit(env, max_episode_steps=1000)
    # env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)
    return env