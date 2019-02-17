import gym


class LogWrapper(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.max_episode_steps = max_episode_steps

        self.ep_rew = 0.
        self.ep_len = 0

    def reset(self):
        self.ep_len, self.ep_rew = 0, 0.
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.ep_rew += rew
        self.ep_len += 1
        if self.ep_len == self.max_episode_steps:
            done = True
        if done:
            info = {'ep_r': self.ep_rew, 'ep_len': self.ep_len}
        return obs, rew, done, info