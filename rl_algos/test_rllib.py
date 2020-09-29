#!usr/bin/env python

import or_gym
from or_gym import utils
import ray
from ray import tune
from ray.rllib import agents
import warnings
# from argparse import ArgumentParser
# from datetime import datetime
import os
import traceback

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

def test_training(self, env_name='InvManagement-v2'):
    rl_config['env'] = env_name
    ray.init()
    try:
        agent = agents.PPO.PPOTrainer(rl_config, )
        trainer = ray_utils.build_trainer(env_config, rl_config)
        trainer.train()
        success = True
    except Exception as e:
        tb = e.__traceback__
        success = False
    ray.shutdown()
    assert success, ''.join(traceback.format_tb(tb))