'''Entropy (Hyle 7) - PufferLib Ocean Environment

An asymmetric two-player abstract strategy board game by Eric Solomon.
One player is Order (creating palindromic patterns), the other is Chaos
(preventing them). A single RL agent learns both roles.
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.entropy import binding


class Entropy(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
            difficulty=1, agent_role=-1,
            reward_invalid=-0.1, reward_palindrome_delta=0.0,
            width=800, height=600,
            buf=None, seed=0):

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(648,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(246)

        super().__init__(buf=buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            difficulty=difficulty, agent_role=agent_role,
            reward_invalid=reward_invalid,
            reward_palindrome_delta=reward_palindrome_delta,
            width=width, height=height)

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed or 0)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    num_envs = 1000
    env = Entropy(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')


if __name__ == '__main__':
    test_performance()
