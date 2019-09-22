import random

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from loguru import logger as log


class MatchInputContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.target = None
        self._max_episode_steps = 10**3
        self.step_num = 0

    def step(self, action: np.ndarray):
        action = action[0]
        reward = 1 - abs(self.target - action)
        # log.info(f'reward={reward}')
        # log.info(f'action={action}')
        # log.info(f'target={self.target}')
        done = np.isclose(self.target, action)
        observation = self.get_observation()
        self.step_num += 1
        if self.step_num % 100 == 0:
            # HACK: End episodes to work with pytorch-actor-critic
            done = True
        return observation, reward, done, {}

    def seed(self, seed=None):
        seed = seed or 1
        random.seed(seed)

    def reset(self):
        self.step_num = 0
        self.target = random.random()
        return self.get_observation()

    def get_observation(self):
        return np.array([self.target])

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class MatchPairsContinuousEnv(MatchInputContinuousEnv):
    def __init__(self):
        super(MatchPairsContinuousEnv, self).__init__()
        self.pairs = [(-.1, .8), (.44, .55), (-.8, -.79), (.99, 0), (0, 0)]

    def reset(self):
        self.step_num = 0
        return self.get_observation()

    def get_observation(self):
        index = random.randint(0, len(self.pairs) - 1)
        pair = self.pairs[index]

        # TODO: Pair sequence
        # pair = self.pairs[self.step_num % len(self.pairs)]

        obs = pair[0]
        self.target = pair[1]
        return np.array([obs])


class CorrectivePsuedoSteeringEnv(MatchInputContinuousEnv):
    """Reverse direction env (like a corrective velocity)"""
    def __init__(self):
        super(CorrectivePsuedoSteeringEnv, self).__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,))
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
        self.action_range = \
            (self.action_space.high - self.action_space.low)[0]


    def reset(self):
        self.step_num = 0
        return self.get_observation()

    def get_observation(self):
        # Get random between -10 and 10 for lane deviation
        lane_deviation = random.random() * 20 - 10

        # TODO: If longitudinal, get speed between 0 and 10

        # Target is then opposite of lane deviation
        self.target = -lane_deviation

        return np.array([lane_deviation])

    def step(self, action: np.ndarray):
        action = action[0]
        # Avoid negative rewards due to
        # https://github.com/deepdrive/deepdrive/blob/9cf954fd55e085b72e7a217fb02d76189998d6d0/vendor/openai/baselines/ppo2/ppo2.py#L108-L118

        delta = abs(self.target - action)

        # Normalize reward to between 0 and 1
        reward = (self.action_range - delta) / self.action_range
        observation = self.get_observation()
        self.step_num += 1

        # HACK: End episodes to work with pytorch-actor-critic
        done = (self.step_num % 100 == 0)

        return observation, reward, done, {}


class MatchInputSequenceContinuousEnv(MatchInputContinuousEnv):
    def __init__(self):
        super(MatchInputSequenceContinuousEnv, self).__init__()

