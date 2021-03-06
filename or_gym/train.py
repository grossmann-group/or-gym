import or_gym
from or_gym.utils import create_env
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import ray
from ray import tune

env_name = 'InvManagement-v3'
algo = 'PPO'
env_config = {}

rl_config = dict(
    env=env_name,
    num_workers=3,
    env_config=env_config,
    model=dict(
        vf_share_layers=False,
        fcnet_activation='elu',
        fcnet_hiddens=[256, 256]
    ),
    lr=tune.grid_search([1e-5, 1e-4, 1e-6]),
    vf_loss_coeff=tune.grid_search([1.0, 0.5, 0.1])
)

stop = {
    'training_iteration': 500
}

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))

register_env(env_name, env_config=env_config)

# try:
results = tune.run(
    algo,
    config=rl_config,
    stop=stop,
    checkpoint_freq=100,
    checkpoint_at_end=True,
    reuse_actors=True,
    # keep_num_checkpoints=3, # Limits number of checkpoints
    checkpoint_score_attr='episode_reward_mean'
    # resources_per_trial={"cpu": 4, "gpu": 0},
)
# except Exception as e:
#     pass

df = results.dataframe()
df.to_csv(f'RESULTS/{env_name}_results.csv', index_label=False)
