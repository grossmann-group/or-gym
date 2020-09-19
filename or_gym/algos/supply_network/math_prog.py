#!usr/bin/env python

import or_gym
import pyomo.environ as pe
from pyomo.core.util import prod
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *

def net_im_pi_lp_model(env,use_expectation=False):
    '''
    Build a perfect information LP model (Oracle) for the InvManagement problem (v2 and v3).
    No policy is used for the reorder. This will give you the optimal reorder quantities if 
    you knew the demand before hand.
    ''' 
    
    #create model
    lp = pe.ConcreteModel()
    
    #define sets
    lp.J = pe.Set(initialize=env.main_nodes)
    lp.Jraw = pe.Set(initialize=env.rawmat)
    lp.Jdistrib = pe.Set(initialize=env.distrib)
    lp.Jretail = pe.Set(initialize=env.retail)
    lp.Jmarket = pe.Set(initialize=env.market)
    lp.Jfactory = pe.Set(initialize=env.factory)
    lp.network_links = pe.Set(dimen=2, initialize=env.network_links)
    lp.reorder_links = pe.Set(dimen=2, initialize=env.reorder_links)
    lp.retail_links = pe.Set(dimen=2, initialize=env.retail_links)
    lp.demands = pe.Set(dimen=3, initialize=[(t,) + e for e in env.retail_links for t in range(env.num_periods)])
    lp.T = pe.RangeSet(0,env.num_periods-1) 
    lp.T1 = pe.RangeSet(0,env.num_periods)
    
    #define parameters
    lp.h = pe.Param(lp.J, initialize = {j:env.graph.nodes[j]['h'] for j in lp.J}) #inventory holding cost at main nodes
    lp.C = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['C'] for j in lp.Jfactory}) #production capacity at each factory node
    lp.o = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['o'] for j in lp.Jfactory}) #operating cost at each factory node
    lp.v = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['v'] for j in lp.Jfactory}) #production yield at each factory node
    lp.p = pe.Param(lp.network_links, initialize = {e:env.graph.edges[e]['p'] for e in lp.network_links}) #price for selling/purchasing on a link
    lp.L = pe.Param(lp.reorder_links, initialize = {e:env.graph.edges[e]['L'] for e in lp.reorder_links}) #price for selling/purchasing on a link
    lp.g = pe.Param(lp.reorder_links, initialize = {e:env.graph.edges[e]['g'] for e in lp.reorder_links}) #price for selling/purchasing on a link
    lp.b = pe.Param(lp.retail_links, initialize = {e:env.graph.edges[e]['b'] for e in lp.retail_links}) #price for selling/purchasing on a link
    alpha = env.alpha #time-valued discount 
    num_periods = env.num_periods #number of periods
    backlog = env.backlog #backlog or lost sales
    #demand profile
    D = {e:env.graph.edges[e]['demand_dist'] if isinstance(env.graph.edges[e]['demand_dist'],(list, np.ndarray)) else env.graph.edges[e]['demand_dist'].rvs(size=env.num_periods,**env.graph.edges[e]['dist_param']) for e in lp.retail_links} #demands on a retail link for each period
    lp.D = pe.Param(lp.demands, initialize = {te:D[te[1:]][te[0]] for te in lp.demands}) #store demands
    prob = {e:[list(D[e]).count(D[e][t])/len(D[e]) for t in range(env.num_periods)] if isinstance(env.graph.edges[e]['demand_dist'],(list, np.ndarray)) else env.graph.edges[e]['demand_dist'].pmf(D[e],**env.graph.edges[e]['dist_param']) for e in lp.retail_links} #probability of each demand based on distribution
    lp.prob = pe.Param(lp.demands, initialize = {te:prob[te[1:]][te[0]] for te in lp.demands}) #store probability at each period
    
    #define variables
    lp.X = pe.Var(lp.T1,lp.J,domain=pe.NonNegativeReals) #on hand inventory at each node
    lp.Y = pe.Var(lp.T1,lp.reorder_links,domain=pe.NonNegativeReals) #pipeline inventory on each link
    lp.R = pe.Var(lp.T,lp.reorder_links,domain=pe.NonNegativeReals) #reorder quantities for each node
    lp.S = pe.Var(lp.T,lp.network_links,domain=pe.NonNegativeReals) #sales at each node
    lp.U = pe.Var(lp.T,lp.retail_links,domain=pe.NonNegativeReals) #unfulfilled sales at each node
    lp.P = pe.Var(lp.T,lp.J,domain=pe.Reals) #profit at each node
    
    #initialize on-hand and pipeline inventories
    for j in lp.J:
        lp.X[0,j].fix(env.graph.nodes[j]['I0'])
        for k in env.graph.predecessors(j):
            lp.Y[0,k,j].fix(0)
    
    #define constraints
    lp.profit = pe.ConstraintList()
    lp.inv_bal = pe.ConstraintList()
    lp.pip_bal = pe.ConstraintList()
    lp.reorder1 = pe.ConstraintList()
    lp.reorder2 = pe.ConstraintList()
    lp.reorder3 = pe.ConstraintList()
    lp.sales1 = pe.ConstraintList()
    lp.sales2 = pe.ConstraintList()
    lp.sales3 = pe.ConstraintList()
    lp.unfulfilled = pe.ConstraintList()
    
    #build constraints
    for t in lp.T:
        for j in lp.J:
            #profit
            if j in lp.Jretail:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - sum(lp.b[j,k]*lp.U[t,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     - sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jdistrib:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     + sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jfactory:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - lp.o[j]/lp.v[j]*sum(lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     + sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            #on-hand inventory
            if j in lp.Jdistrib:
                lp.inv_bal.add(lp.X[t+1,j] == lp.X[t,j] 
                                            + sum(lp.R[t-lp.L[k,j],k,j] for k in env.graph.predecessors(j) if t-lp.L[k,j]>=0)
                                            - sum(lp.S[t,j,k] for k in env.graph.successors(j)))
            else:
                lp.inv_bal.add(lp.X[t+1,j] == lp.X[t,j] 
                                            + sum(lp.R[t-lp.L[k,j],k,j] for k in env.graph.predecessors(j) if t-lp.L[k,j]>=0)
                                            - 1/lp.v[j] * sum(lp.S[t,j,k] for k in env.graph.successors(j)))
            #pipeline inventory
            for k in env.graph.predecessors(j):
                if t-lp.L[k,j]<0:
                    lp.pip_bal.add(lp.Y[t+1,k,j] == lp.Y[t,k,j] + lp.R[t,k,j])
                else:
                    lp.pip_bal.add(lp.Y[t+1,k,j] == lp.Y[t,k,j]
                                                    - lp.R[t-lp.L[k,j],k,j]
                                                    + lp.R[t,k,j])

            #reorder quantities
            if j in lp.Jdistrib and j not in lp.Jretail:
                lp.reorder1.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j])
            elif j in lp.Jfactory:
                lp.reorder2.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j]*lp.v[j])
                lp.reorder3.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.C[j])

            #sales quantities
            if j in lp.Jretail:
                lp.sales1.add(sum(lp.S[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j] 
                                                                                 + sum(lp.R[t-lp.L[k,j],k,j] for k in env.graph.predecessors(j) if t-lp.L[k,j]>=0))
                for k in env.graph.successors(j):
                    if not backlog or t == 0:
                        lp.sales2.add(lp.S[t,j,k] <= lp.D[t,j,k])
                    else:
                        lp.sales2.add(lp.S[t,j,k] <= lp.D[t,j,k] + lp.U[t-1,j,k])
            else:
                for k in env.graph.successors(j):
                    lp.sales3.add(lp.S[t,j,k] == lp.R[t,j,k])

            #unfulfilled orders
            if j in lp.Jretail:
                for k in env.graph.successors(j):
                    if not backlog or t == 0:
                        lp.unfulfilled.add(lp.U[t,j,k] == lp.D[t,j,k] - lp.S[t,j,k])
                    else:
                        lp.unfulfilled.add(lp.U[t,j,k] == lp.D[t,j,k] + lp.U[t-1,j,k] - lp.S[t,j,k])

    #objective function: maximize average profit
    if use_expectation:
        lp.obj = pe.Objective(expr = 1/num_periods * sum( prod([lp.prob[t,e[0],e[1]] for e in env.retail_links]) * sum(lp.P[t,j] for j in lp.J) for t in lp.T),
                            sense = pe.maximize)
    else:
        lp.obj = pe.Objective(expr = 1/num_periods * sum(lp.P[t,j] for j in lp.J for t in lp.T),
                            sense = pe.maximize)
    
    return lp

def net_im_lp_model(env, window_size=np.Inf, use_expectation=False):
    '''
    Build a rolling/shrinking horizon LP model for the InvManagement problem (v2 and v3).
    The model assumes the average demand from the specified distribution is used
    For a rolling horizon, specify a window_size; otherwise, a shrinking horizon
    approach is used.
    ''' 
    
    #adjust window_size towards the end of the simulation (shrinking horizon)
    window_size = min(window_size, env.num_periods - env.period)

    #create model
    lp = pe.ConcreteModel()
    
    #define sets
    lp.J = pe.Set(initialize=env.main_nodes)
    lp.Jraw = pe.Set(initialize=env.rawmat)
    lp.Jdistrib = pe.Set(initialize=env.distrib)
    lp.Jretail = pe.Set(initialize=env.retail)
    lp.Jmarket = pe.Set(initialize=env.market)
    lp.Jfactory = pe.Set(initialize=env.factory)
    lp.network_links = pe.Set(dimen=2, initialize=env.network_links)
    lp.reorder_links = pe.Set(dimen=2, initialize=env.reorder_links)
    lp.retail_links = pe.Set(dimen=2, initialize=env.retail_links)
    lp.demands = pe.Set(dimen=3, initialize=[(t,) + e for e in env.retail_links for t in range(window_size)])
    lp.T = pe.RangeSet(0,window_size-1) 
    lp.T1 = pe.RangeSet(0,window_size)
    
    #define parameters
    lp.h = pe.Param(lp.J, initialize = {j:env.graph.nodes[j]['h'] for j in lp.J}) #inventory holding cost at main nodes
    lp.C = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['C'] for j in lp.Jfactory}) #production capacity at each factory node
    lp.o = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['o'] for j in lp.Jfactory}) #operating cost at each factory node
    lp.v = pe.Param(lp.Jfactory, initialize = {j:env.graph.nodes[j]['v'] for j in lp.Jfactory}) #production yield at each factory node
    lp.p = pe.Param(lp.network_links, initialize = {e:env.graph.edges[e]['p'] for e in lp.network_links}) #price for selling/purchasing on a link
    lp.L = pe.Param(lp.reorder_links, initialize = {e:env.graph.edges[e]['L'] for e in lp.reorder_links}) #price for selling/purchasing on a link
    lp.g = pe.Param(lp.reorder_links, initialize = {e:env.graph.edges[e]['g'] for e in lp.reorder_links}) #price for selling/purchasing on a link
    lp.b = pe.Param(lp.retail_links, initialize = {e:env.graph.edges[e]['b'] for e in lp.retail_links}) #price for selling/purchasing on a link
    alpha = env.alpha #time-valued discount 
    backlog = env.backlog #backlog or lost sales
    #demand profile
    D = {e:np.ones(window_size)*np.mean(env.graph.edges[e]['demand_dist']) if isinstance(env.graph.edges[e]['demand_dist'],(list, np.ndarray)) else np.ones(window_size)*env.graph.edges[e]['demand_dist'].mean(**env.graph.edges[e]['dist_param']) for e in lp.retail_links} #demands on a retail link for each period
    lp.D = pe.Param(lp.demands, initialize = {te:D[te[1:]][te[0]] for te in lp.demands}) #store demands
    prob = {e:[1/window_size for t in range(window_size)] if isinstance(env.graph.edges[e]['demand_dist'],(list, np.ndarray)) else env.graph.edges[e]['demand_dist'].pmf(D[e],**env.graph.edges[e]['dist_param']) for e in lp.retail_links} #probability of each demand based on distribution
    lp.prob = pe.Param(lp.demands, initialize = {te:prob[te[1:]][te[0]] for te in lp.demands}) #store probability at each period
    
    #define variables
    lp.X = pe.Var(lp.T1,lp.J,domain=pe.NonNegativeReals) #on hand inventory at each node
    lp.Y = pe.Var(lp.T1,lp.reorder_links,domain=pe.NonNegativeReals) #pipeline inventory on each link
    lp.R = pe.Var(lp.T,lp.reorder_links,domain=pe.NonNegativeReals) #reorder quantities for each node
    lp.S = pe.Var(lp.T,lp.network_links,domain=pe.NonNegativeReals) #sales at each node
    lp.U = pe.Var(lp.T,lp.retail_links,domain=pe.NonNegativeReals) #unfulfilled sales at each node
    lp.P = pe.Var(lp.T,lp.J,domain=pe.Reals) #profit at each node
    
    #initialize on-hand and pipeline inventories
    for j in lp.J:
        lp.X[0,j].fix(env.X.loc[env.period,j])
        for k in env.graph.predecessors(j):
            lp.Y[0,k,j].fix(env.Y.loc[env.period,(k,j)])
    
    #define constraints
    lp.profit = pe.ConstraintList()
    lp.inv_bal = pe.ConstraintList()
    lp.pip_bal = pe.ConstraintList()
    lp.reorder1 = pe.ConstraintList()
    lp.reorder2 = pe.ConstraintList()
    lp.reorder3 = pe.ConstraintList()
    lp.sales1 = pe.ConstraintList()
    lp.sales2 = pe.ConstraintList()
    lp.sales3 = pe.ConstraintList()
    lp.unfulfilled = pe.ConstraintList()
    
    #build constraints
    for t in lp.T:
        for j in lp.J:
            #profit
            if j in lp.Jretail:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - sum(lp.b[j,k]*lp.U[t,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     - sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jdistrib:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     + sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jfactory:
                lp.profit.add(lp.P[t,j] == alpha**t * (sum(lp.p[j,k]*lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[t,k,j] for k in env.graph.predecessors(j))
                                                     - lp.o[j]/lp.v[j]*sum(lp.S[t,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[t+1,j] 
                                                     + sum(lp.g[k,j]*lp.Y[t+1,k,j] for k in env.graph.predecessors(j))))
            #on-hand inventory
            if j in lp.Jdistrib:
                lp.inv_bal.add(lp.X[t+1,j] == lp.X[t,j] 
                                            + sum(0 if env.period+t-lp.L[k,j] < 0 else 
                                                        lp.R[t-lp.L[k,j],k,j] if t-lp.L[k,j] >= 0 else env.R.loc[env.period+t-lp.L[k,j],(k,j)] 
                                                            for k in env.graph.predecessors(j))
                                            - sum(lp.S[t,j,k] for k in env.graph.successors(j)))
            else:
                lp.inv_bal.add(lp.X[t+1,j] == lp.X[t,j] 
                                            + sum(0 if env.period+t-lp.L[k,j] < 0 else 
                                                        lp.R[t-lp.L[k,j],k,j] if t-lp.L[k,j] >= 0 else env.R.loc[env.period+t-lp.L[k,j],(k,j)] 
                                                            for k in env.graph.predecessors(j))
                                            - 1/lp.v[j] * sum(lp.S[t,j,k] for k in env.graph.successors(j)))
            #pipeline inventory
            for k in env.graph.predecessors(j):
                if env.period+t-lp.L[k,j] < 0: #if reorder is prior to when the problem started
                    lp.pip_bal.add(lp.Y[t+1,k,j] == lp.Y[t,k,j] + lp.R[t,k,j])
                elif t-lp.L[k,j]>=0: #if reorder is in the current horizon scope
                    lp.pip_bal.add(lp.Y[t+1,k,j] == lp.Y[t,k,j]
                                                    - lp.R[t-lp.L[k,j],k,j]
                                                    + lp.R[t,k,j])
                else: #if reorder is available in history
                    lp.pip_bal.add(lp.Y[t+1,k,j] == lp.Y[t,k,j]
                                                    - env.R.loc[env.period+t-lp.L[k,j],(k,j)]
                                                    + lp.R[t,k,j])

            #reorder quantities
            if j in lp.Jdistrib and j not in lp.Jretail:
                lp.reorder1.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j])
            elif j in lp.Jfactory:
                lp.reorder2.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j]*lp.v[j])
                lp.reorder3.add(sum(lp.R[t,j,k] for k in env.graph.successors(j)) <= lp.C[j])

            #sales quantities
            if j in lp.Jretail:
                lp.sales1.add(sum(lp.S[t,j,k] for k in env.graph.successors(j)) <= lp.X[t,j] 
                                                                                    + sum(0 if env.period+t-lp.L[k,j] < 0 else 
                                                                                                lp.R[t-lp.L[k,j],k,j] if t-lp.L[k,j] >= 0 else env.R.loc[env.period+t-lp.L[k,j],(k,j)] 
                                                                                                    for k in env.graph.predecessors(j)))
                for k in env.graph.successors(j):
                    if not backlog or env.period+t-1 < 0: #if lost sales or if prevoius period is before the problem started
                        lp.sales2.add(lp.S[t,j,k] <= lp.D[t,j,k])
                    elif t-1 >= 0: #if backlog quanity from previous period is known
                        lp.sales2.add(lp.S[t,j,k] <= lp.D[t,j,k] + lp.U[t-1,j,k])
                    else: #if backlog quanitty is available in history
                        lp.sales2.add(lp.S[t,j,k] <= lp.D[t,j,k] + env.U.loc[env.period+t-1,(j,k)])
            else:
                for k in env.graph.successors(j):
                    lp.sales3.add(lp.S[t,j,k] == lp.R[t,j,k])

            #unfulfilled orders
            if j in lp.Jretail:
                for k in env.graph.successors(j):
                    if not backlog or env.period+t-1 < 0:
                        lp.unfulfilled.add(lp.U[t,j,k] == lp.D[t,j,k] - lp.S[t,j,k])
                    elif t-1 >= 0:
                        lp.unfulfilled.add(lp.U[t,j,k] == lp.D[t,j,k] + lp.U[t-1,j,k] - lp.S[t,j,k])
                    else:
                        lp.unfulfilled.add(lp.U[t,j,k] == lp.D[t,j,k] + env.U.loc[env.period+t-1,(j,k)] - lp.S[t,j,k])

    #objective function: maximize average profit
    if use_expectation:
        lp.obj = pe.Objective(expr = 1/(window_size) * sum( prod([lp.prob[t,e[0],e[1]] for e in env.retail_links]) * sum(lp.P[t,j] for j in lp.J) for t in lp.T),
                            sense = pe.maximize)
    else:
        lp.obj = pe.Objective(expr = 1/(window_size) * sum(lp.P[t,j] for j in lp.J for t in lp.T),
                            sense = pe.maximize)
    
    return lp