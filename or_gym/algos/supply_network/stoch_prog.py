#!usr/bin/env python

import or_gym
import pyomo.environ as pe
from pyomo.core.util import prod
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *
import math


def net_im_stoch_lp_model(env, window_size=np.Inf, perfect_information = False, use_expectation=False):
    base_values = {"high":25, "low":15, "medium":20}
    mu = 20
    base_prob = {"high":1-poisson.cdf(22, mu), "low":poisson.cdf(17, mu), "medium":(poisson.cdf(22, mu)-poisson.cdf(17, mu))}

    #we assume the scenario tree only grows for max_scenario_depth and then it starts to take the mean value of the demand
    max_scenario_tree_depth = 6
    window_size = min(window_size, env.num_periods - env.period)
    #we represent a node as (t, node_number) for example, the root node is represented as (0,0)
    G = nx.DiGraph()
    G.add_node((0,0), prob=1, demand=0) #add root node 
    # G.nodes(data=True)[(0,0)]
    cur_num_nodes = 0 
    leaf_nodes = [(0,0)]
    nstages = min(window_size, max_scenario_tree_depth)
    #TODO check the terminal condition
    for t in range(1, nstages):
        new_leaf_nodes = []
        for n in leaf_nodes:
            for new_node in ["high", "low", "medium"]:
                G.add_node((t,cur_num_nodes+1), prob=G.nodes(data=True)[n]["prob"]*base_prob[new_node], demand=base_values[new_node])
                new_leaf_nodes.append((t,cur_num_nodes+1))
                G.add_edge(n, (t,cur_num_nodes+1))
                cur_num_nodes += 1
        leaf_nodes = new_leaf_nodes

    #construct the nodes with mean values after nstages
    if nstages < window_size+1:
        for t in range(nstages, window_size+1):
            new_leaf_nodes = []
            for n in leaf_nodes:
                G.add_node((t,cur_num_nodes+1), prob=G.nodes(data=True)[n]["prob"], demand=20)
                new_leaf_nodes.append((t,cur_num_nodes+1))
                G.add_edge(n, (t,cur_num_nodes+1))
                cur_num_nodes += 1   
            leaf_nodes = new_leaf_nodes     


    '''
    Build an LP model for the supply chain network InvManagement problem (v2 and v3).
    Three modes exist:
        1) Perfect information (Oracle): Gives the optimal reorder quantities if 
            the demand were known before hand. Set perfect_information=True.
        2) Shrinking horizon: Assumes the average demand from the specified distribution 
            is used. Set window_size=np.Inf
        3) Rolling horizon: Assumes the average demand from the specified distribution 
            is used. Set window_size for the rolling window.
    ''' 

    #adjust window_size towards the end of the simulation (shrinking horizon)
    


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
    lp.Nodes = pe.Set(initialize=list(G.nodes))
    lp.nonLeafNodes = pe.Set(initialize=list(set(G.nodes) - set(leaf_nodes)))
    nodes = list(G.nodes)
    nodes.remove((0,0))
    lp.nonRootNodes = pe.Set(initialize=nodes)
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

    #define variables
    lp.X = pe.Var(lp.Nodes,lp.J,domain=pe.NonNegativeReals) #on hand inventory at each node
    lp.Y = pe.Var(lp.Nodes,lp.reorder_links,domain=pe.NonNegativeReals) #pipeline inventory on each link
    lp.R = pe.Var(lp.nonLeafNodes,lp.reorder_links,domain=pe.NonNegativeReals) #reorder quantities for each node
    lp.S = pe.Var(lp.nonRootNodes,lp.network_links,domain=pe.NonNegativeReals) #sales at each node
    lp.U = pe.Var(lp.nonRootNodes,lp.retail_links,domain=pe.NonNegativeReals) #unfulfilled sales at each node
    lp.P = pe.Var(lp.nonRootNodes,lp.J,domain=pe.Reals) #profit at each node

    #initialize on-hand and pipeline inventories
    for j in lp.J:
        lp.X[(0,0),j].fix(env.X.loc[env.period,j])
        for k in env.graph.predecessors(j):
            lp.Y[(0,0),k,j].fix(env.Y.loc[env.period,(k,j)])

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
    for n in lp.nonRootNodes:
        for j in lp.J:
            t = n[0]
            path_to_root = []
            cur_node = n
            path_to_root.append(cur_node)
            for p in range(100):
                if len(G.pred[cur_node]) == 1:
                    cur_node = list(G.pred[cur_node])[0]
                    path_to_root.append(cur_node)
                else:
                    break 


            #profit
            if j in lp.Jretail:
                lp.profit.add(lp.P[n,j] == alpha**(t-1) * (sum(lp.p[j,k]*lp.S[n,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[path_to_root[1],k,j] for k in env.graph.predecessors(j))
                                                     - sum(lp.b[j,k]*lp.U[n,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[n,j] 
                                                     - sum(lp.g[k,j]*lp.Y[n,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jdistrib:
                lp.profit.add(lp.P[n,j] == alpha**(t-1) * (sum(lp.p[j,k]*lp.S[n,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[path_to_root[1],k,j] for k in env.graph.predecessors(j))
                                                     - lp.h[j]*lp.X[n,j] 
                                                     + sum(lp.g[k,j]*lp.Y[n,k,j] for k in env.graph.predecessors(j))))
            elif j in lp.Jfactory:
                lp.profit.add(lp.P[n,j] == alpha**(t-1) * (sum(lp.p[j,k]*lp.S[n,j,k] for k in env.graph.successors(j))
                                                     - sum(lp.p[k,j]*lp.R[path_to_root[1],k,j] for k in env.graph.predecessors(j))
                                                     - lp.o[j]/lp.v[j]*sum(lp.S[n,j,k] for k in env.graph.successors(j))
                                                     - lp.h[j]*lp.X[n,j] 
                                                     + sum(lp.g[k,j]*lp.Y[n,k,j] for k in env.graph.predecessors(j))))
            #on-hand inventory
            if j in lp.Jdistrib:
                lp.inv_bal.add(lp.X[n,j] == lp.X[path_to_root[1],j] 
                                            + sum(0 if env.period+t-lp.L[k,j]-1 < 0 else 
                                                        lp.R[path_to_root[lp.L[k,j]+1],k,j] if t-lp.L[k,j]-1 >= 0 else env.R.loc[env.period+t-lp.L[k,j]-1,(k,j)] 
                                                            for k in env.graph.predecessors(j))
                                            - sum(lp.S[n,j,k] for k in env.graph.successors(j)))
            else:
                lp.inv_bal.add(lp.X[n,j] == lp.X[path_to_root[1],j] 
                                            + sum(0 if env.period+t-lp.L[k,j]-1 < 0 else 
                                                        lp.R[path_to_root[lp.L[k,j]+1],k,j] if t-lp.L[k,j]-1 >= 0 else env.R.loc[env.period+t-lp.L[k,j]-1,(k,j)] 
                                                            for k in env.graph.predecessors(j))
                                            - 1/lp.v[j] * sum(lp.S[n,j,k] for k in env.graph.successors(j)))
            #pipeline inventory
            for k in env.graph.predecessors(j):
                if env.period+t-lp.L[k,j]-1 < 0: #if reorder is prior to when the problem started
                    lp.pip_bal.add(lp.Y[n,k,j] == lp.Y[path_to_root[1],k,j] + lp.R[path_to_root[1],k,j])
                elif t-lp.L[k,j]-1>=0: #if reorder is in the current horizon scope
                    lp.pip_bal.add(lp.Y[n,k,j] == lp.Y[path_to_root[1],k,j]
                                                    - lp.R[path_to_root[lp.L[k,j]+1],k,j]
                                                    + lp.R[path_to_root[1],k,j])
                else: #if reorder is available in history
                    lp.pip_bal.add(lp.Y[n,k,j] == lp.Y[path_to_root[1],k,j]
                                                    - env.R.loc[env.period+t-lp.L[k,j]-1,(k,j)]
                                                    + lp.R[path_to_root[1],k,j])

            #reorder quantities
            if j in lp.Jdistrib and j not in lp.Jretail:
                lp.reorder1.add(sum(lp.R[path_to_root[1],j,k] for k in env.graph.successors(j)) <= lp.X[path_to_root[1],j])
            elif j in lp.Jfactory:
                lp.reorder2.add(sum(lp.R[path_to_root[1],j,k] for k in env.graph.successors(j)) <= lp.X[path_to_root[1],j]*lp.v[j])
                lp.reorder3.add(sum(lp.R[path_to_root[1],j,k] for k in env.graph.successors(j)) <= lp.C[j])

            #sales quantities
            if j in lp.Jretail:
                lp.sales1.add(sum(lp.S[n,j,k] for k in env.graph.successors(j)) <= lp.X[path_to_root[1],j] 
                                                                                    + sum(0 if env.period+t-lp.L[k,j]-1 < 0 else 
                                                                                                lp.R[path_to_root[lp.L[k,j]+1],k,j] if t-lp.L[k,j]-1 >= 0 else env.R.loc[env.period+t-lp.L[k,j]-1,(k,j)] 
                                                                                                    for k in env.graph.predecessors(j)))
                for k in env.graph.successors(j):
                    if not backlog or env.period+t-2 < 0: #if lost sales or if prevoius period is before the problem started
                        lp.sales2.add(lp.S[n,j,k] <= G.nodes(data=True)[n]["demand"])
                    elif t-2 >= 0: #if backlog quanity from previous period is known
                        lp.sales2.add(lp.S[n,j,k] <= G.nodes(data=True)[n]["demand"] + lp.U[path_to_root[1],j,k])
                    else: #if backlog quanitty is available in history
                        lp.sales2.add(lp.S[n,j,k] <= G.nodes(data=True)[n]["demand"] + env.U.loc[env.period+t-2,(j,k)])
            else:
                for k in env.graph.successors(j):
                    lp.sales3.add(lp.S[n,j,k] == lp.R[path_to_root[1],j,k])

            #unfulfilled orders
            if j in lp.Jretail:
                for k in env.graph.successors(j):
                    if not backlog or env.period+t-2 < 0:
                        lp.unfulfilled.add(lp.U[n,j,k] == G.nodes(data=True)[n]["demand"] - lp.S[n,j,k])
                    elif t-2 >= 0:
                        lp.unfulfilled.add(lp.U[n,j,k] == G.nodes(data=True)[n]["demand"] + lp.U[path_to_root[1],j,k] - lp.S[n,j,k])
                    else:
                        lp.unfulfilled.add(lp.U[n,j,k] == G.nodes(data=True)[n]["demand"] + env.U.loc[env.period+t-2,(j,k)] - lp.S[n,j,k])

    #objective function: maximize average profit

    lp.obj = pe.Objective(expr = 1/(window_size) * sum(  sum(lp.P[n,j] * G.nodes(data=True)[n]["prob"] for j in lp.J) for n in lp.nonRootNodes),
                            sense = pe.maximize)

    return lp 



