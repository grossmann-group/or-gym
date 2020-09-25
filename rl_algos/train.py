#!usr/bin/env python

import or_gym
from or_gym import utils
import ray
from ray import tune
import warnings
from argparse import ArgumentParser
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

rl_config = {
 'num_workers': 4,
 'num_envs_per_worker': 1,
 'train_batch_size': 4000,
 'lr': tune.grid_search([1e-3, 1e-4, 1e-5]),
 'gamma': tune.grid_search([1, 0.99, 0.9]),
 'model': {
     'fcnet_activation': 'elu',
     'fcnet_hiddens': [256, 256, 256],
     'vf_share_layers': False
 }
}

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default='InvManagement-v2')
    # parser.add_argument('--env_config', type=str, default='')
    parser.add_argument('--mask', type=bool, default=False)
    # parser.add_argument('--rl_config', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--results_dir', type=str, default='./logs')

    return parser.parse_args()

def register_env(env_name, env_config={}, model_name=None):
    env = utils.create_env(env_name)
    tune.register_env(env_name, lambda env_name: env(env_name, env_config=env_config))
    if model_name is None:
        model_name = env_name
    return model_name

def tune_model(rl_config, args, env_config={}, model_name=None):
    env_name = rl_config['env']
    model_name = register_env(env_name, env_config=env_config, model_name=model_name)
    ray.init()
    results = tune.run(
        args.algo.upper(),
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        queue_trials=True,
        stop={
            'training_iteration': 500,
        },
        config=rl_config
    )
    timestamp = datetime.strftime(datetime.today(), '%Y-%m-%d')
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    results_path = os.path.join(args.results_dir, model_name + '_' + timestamp + '.csv')
    results.dataframe().to_csv(results_path)

if __name__ == '__main__':
    args = parse_arguments()
    env_name = args.env_name
    rl_config['env'] = env_name
    tune_model(rl_config, args)