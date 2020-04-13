import os

from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch
from gym_match_input_continuous.experiments import utils

experiment_name = os.path.basename(__file__)[:-3]
notes = """
Garbage into RL garbage out
"""


env_config = dict(
    env_name='match-input-continuous-v0',
)

net_config = dict(
    hidden_units=(32, 32),
    activation=torch.nn.Tanh
)

eg = ExperimentGrid(name=experiment_name)
eg.add('env_name', env_config['env_name'], '', False)
# eg.add('gamma', 0.999)  # Lower gamma so seconds of effective horizon remains at 10s with current physics steps = 12 * 1/60s * 1 / (1-gamma)
eg.add('epochs', 1000)
eg.add('steps_per_epoch', 500)
eg.add('try_rollouts', 2)
eg.add('take_worst_rollout', True)
eg.add('steps_per_try_rollout', 1)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)