import or_gym
import numpy as np

env_name = 'InvManagement-v2'

env = or_gym.make(env_name)
action = env.action_space.sample()
env.step(action)
# z = np.ones(env.num_stages-1)

# print(env.base_stock_action(z))