from random import random

import gym


def main():
    env = gym.make('gym_match_input_continuous:match-input-continuous-v0')
    while True:
        ob, reward, done, info = env.step([random()])


if __name__ == '__main__':
    main()
