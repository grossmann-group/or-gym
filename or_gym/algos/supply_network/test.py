import or_gym
import numpy as np 
import pandas as pd 
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *

#solve perfect information model
env1=or_gym.make("InvManagement-v2")
m1=im_pi_lp_model(env1)
s1=SolverFactory('gurobi')
res1=s1.solve(m1)

#solve shrinking horizon model at t=0
env3=or_gym.make("InvManagement-v2")
m3=im_sh_lp_model(env3)
s3=SolverFactory('gurobi')
res3=s3.solve(m3)

#solve perfect information model with average demand
env4=or_gym.make("InvManagement-v2")
env4.graph.edges[(1,0)]['demand_dist']=[20 for i in range(env4.num_periods)]
m4=im_pi_lp_model(env4)
s4=SolverFactory('gurobi')
res4=s4.solve(m4)

#solve shrinking horizon model
env2=or_gym.make("InvManagement-v2")
for t in range(env2.num_periods):
    m2=im_sh_lp_model(env2)
    s2=SolverFactory('gurobi')
    res2=s2.solve(m2)
    Ropt=m2.R.get_values()
    action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
    env2.step(action)

#solve rolling horizon model with window = 10
env5=or_gym.make("InvManagement-v2")
for t in range(env5.num_periods):
    m5=im_rh_lp_model(env5,10)
    s5=SolverFactory('gurobi')
    res5=s5.solve(m5)
    Ropt=m5.R.get_values()
    action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
    env5.step(action)

#solve rolling horizon model with window = 30
env6=or_gym.make("InvManagement-v2")
for t in range(env6.num_periods):
    m6=im_rh_lp_model(env6,30)
    s6=SolverFactory('gurobi')
    res6=s6.solve(m6)
    Ropt=m6.R.get_values()
    action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
    env6.step(action)

#show final total profits
print(np.sum(list(m1.P.get_values().values())))
print(np.sum(list(m4.P.get_values().values())))
print(np.sum(list(m3.P.get_values().values())))
print(env2.P.sum().sum())
print(env5.P.sum().sum())