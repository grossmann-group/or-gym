#!usr/bin/env python

import or_gym
from or_gym.utils import create_env
import ray
from ray import tune
from ray.rllib import agents


env_name = 'InvManagement-v2'
env_config = {}
tune.register_env(env_name, 
    lambda env_name: env(env_name, env_config=env_config))

rl_config = {'num_workers': 1}

ray.init()
trainer = agents.ppo.PPOTrainer(rl_config, env=create_env(env_name))
trainer.train()
