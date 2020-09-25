import or_gym

env_name = 'InvManagement-v2'
env = or_gym.make(env_name)
print(env.step(env.action_space.sample()))
print(env.state.shape[0]==env.obs_dim)