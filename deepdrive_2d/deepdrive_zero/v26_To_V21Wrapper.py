import gymnasium as gym


class V26toV21Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, info

    def reset(self):
        observation, info = self.env.reset()
        return observation
