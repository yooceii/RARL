import gym
import numpy as np

class GVGAIWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        game_id = env.spec.id.split("-")
        game_id[2] = "lvl{}"
        self.game_id = "-".join(game_id)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
            
        if done:
            if info.get("winner") == 3:
                info["episode"]['c'] = 1
            else:
                info["episode"]['c'] = 0
        
        return obs, reward, done, info
    
    def reset(self):
        return self.env.reset(environment_id=self.game_id.format(np.random.randint(0,5)))