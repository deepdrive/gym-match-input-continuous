import random
import sys
from math import exp

import gym
import numpy as np
from box import Box
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
        reward = 1 - abs(self.target - action)  # Max ~1, min ~-1
        # log.info(f'reward={reward}')
        # log.info(f'action={action}')
        # log.info(f'target={self.target}')
        observation = self.get_observation()
        self.step_num += 1
        done = False
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

    def get_state(self):
        return (self.step_num,)

    def set_state(self, s):
        (self.step_num,) = s

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


class StableTrackingEnv(MatchInputContinuousEnv):
    def __init__(self):
        super(StableTrackingEnv, self).__init__()
        self.feed_prev_action = '--zero-prev-action' not in sys.argv
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))
        self.action_range = \
            (self.action_space.high - self.action_space.low)[0]

        assert self.action_range == 1

        # Note, this ends up being ~0.14 after normalization
        self.max_deviation = 0.2  # in both directions (plus / minus)
        self.max_desired_action_change = 0.1

        self.target_min = 0 - self.max_deviation
        self.target_max = 1 + self.max_deviation
        self.prev_action = 0

        # Set True if action scale is not 0=>1
        self.squash_stability_reward = False

    def step(self, action: np.ndarray):
        info = Box(default_box=True)
        action = action[0]

        if not self.prev_action:
            # Technically we could output zero for an action
            # and spuriously trigger this, but very unlikely I think,
            # unless actions are clipped instead of scaled. We should think
            # of a better way to bootstrap prev action...
            reward = 0
        else:
            # Stability is a simplification of minimizing g-force
            stability_reward = self.get_stability_reward(action)

            # Accuracy is a simplification of minimizing lane deviation
            accuracy_reward = 1 - abs(self.target - action)  # 0=>1

            reward = (accuracy_reward + stability_reward) / 2


            stable_action_optimality = self.get_action_optimality(action)

            info.tfx.stable_action_optimality = stable_action_optimality
            info.tfx.stability_reward = stability_reward
            info.tfx.accuracy_reward = accuracy_reward

        # log.info(f'reward={reward}')
        # log.info(f'action={action}')
        # log.info(f'target={self.target}')

        observation = self.get_observation()
        self.step_num += 1
        if self.step_num % 100 == 0:
            # HACK: End episodes to work with pytorch-actor-critic
            done = True
        else:
            done = False
        self.prev_action = action

        return observation, reward, done, info.to_dict()

    def get_stability_reward(self, action):
        mdac = self.max_desired_action_change
        action_change = abs(action - self.prev_action)
        action_overage = max(0, action_change - mdac)
        stability_reward = 1 - action_overage
        if self.squash_stability_reward:
            # Logistic squash from 0=>1
            stability_reward = 1 / (1 + exp(-1 * (stability_reward - 0.5)))
        assert 0 <= stability_reward <= 1
        return stability_reward

    def get_action_optimality(self, action):
        mdac = self.max_desired_action_change
        action_gap = abs(self.target - self.prev_action)
        if action_gap <= mdac:
            optimal_action = self.target
        elif self.target < self.prev_action:
            optimal_action = self.prev_action - mdac
        elif self.target > self.prev_action:
            optimal_action = self.prev_action + mdac
        else:
            raise RuntimeError('Unexpected case')
        action_error = abs(action - optimal_action)
        action_accuracy = 1 - action_error / self.action_range
        return action_accuracy

    def seed(self, seed=None):
        seed = seed or 1
        random.seed(seed)

    def reset(self):
        self.step_num = 0
        self.target = 0
        self.prev_action = 0
        return self.get_observation()

    def get_observation(self):
        md = self.max_deviation
        deviation = md - 2 * md * random.random()
        t = self.target + deviation
        tmin = self.target_min
        tmax = self.target_max

        # Normalize - note this shrinks the max deviation by 1 / (tmax - tmin)
        # TODO: Allow negative targets, i.e. 1 - 2 * below
        self.target = (t - tmin) / (tmax - tmin)

        if self.feed_prev_action:
            prev_action_input = self.prev_action
        else:
            prev_action_input = 0


        return np.array([self.target, prev_action_input])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# TODO: Tracking Env - Slightly change target input, i.e. from 0.1 to 0.2
#  and give reward for not changing output too much - i.e. 0.1 - 0.15 gives
#  max reward. This emulates the trade-off between speed and g-force in driving.
