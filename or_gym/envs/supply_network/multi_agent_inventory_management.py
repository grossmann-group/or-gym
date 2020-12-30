'''
Multi-agent inventory management
Hector Perez, Christian Hubbs, Can Li
12/22/2020
'''

import gym
from gym import spaces
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *
from or_gym.envs.supply_network.inventory_management import NetInvMgmtMasterEnv
from or_gym.utils import assign_env_config
from collections import deque
import matplotlib.pyplot as plt

class MultiAgentNetInvMgmtEnv(NetInvMgmtMasterEnv):
    '''
    This is a multi-agent network optimization environment that is able to explore
    a variety of different models to see how they interact with one another.
    
    In this base case, each node can only submit orders to other nodes further
    upstream.
    Other cases will allow agents to partially fulfill or reject orders as well
    as put in orders and will allow for communication.
    
    Observations are limited to pipelines along connected edges, e.g. agents can
    observe what orders they have placed and sent.
    
    Nodes:
        node_id = Unique identifier for each node
        algo_class = Determines the broad class for each algorithm 
            (RL for reinforcement learning, MILP, DFO, etc.).
        algo = Some classes have multiple algorithms, we use this to differentiate
            between different approaches (e.g. DQN vs PPO).
        SC = Supply capacity
        IC = Inventory capacity
        I0 = initial inventory
        L = lead time for each edge.
        H = holding cost
        B = backlog cost
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(self, *args, **kwargs)
        
        self.agent_ids = np.arange(1, self.num_nodes)
        
        # Disable this for now and let all nodes be RL agents
        self.algo_dict = {
            1: ['RL', 'PPO'],
            2: ['Heuristic', 'ReorderQTY'],
            3: ['RL', 'PPO'],
            4: ['Heuristic', 'ReorderQty'],
            5: ['Heuristic', 'ReorderQty'],
            6: ['Heuristic', 'ReorderQty'],
            7: ['Heuristic', 'ReorderQty'],
            8: ['Heuristic', 'ReorderQty']
        }
        
        assert np.allclose(self.agent_ids, np.array(list(self.algo_dict.keys()))), \
            "Agent ID's and Keys don't match."

        self.retail_supply_nodes = [e[0] for e in self.graph.edges() if 'L' not in self.graph.edges[e]]
            
        self._build_action_and_observation_spaces()
        self.reset()
            
    def _build_action_and_observation_spaces(self):
        # Set up action space and observation space for RL and
        # other models. 
        # RL needs to interact via Ray, so use standard observation and 
        # action space nomenclature. For other models, actions/obs will be
        # stored in tot_act/obs_space.
        # act/obs_dicts provide descriptive mapping of actions to nodes
        self.tot_act_space, action_space, self.act_dict = {}, {}, {}
        self.tot_obs_space, observation_space, self.obs_dict = {}, {}, {}
        
        for i in self.agent_ids:
            self.obs_dict[i] = {}
            self.obs_dict[i][0] = 'Inventory on hand'
            _obs = 1 # Inventory on hand
            _act = 0
            for e in self.graph.edges():
                if i == e[0]:
                    # Downstream connection pipeline and demand
                    L = self.graph.edges[e]['L'] + 1 if 'L' in self.graph.edges[e] else 1
                    _obs += L
                    last_entry = max(list(self.obs_dict[i].keys())) + 1
                    _new_obs = {last_entry + l: f'{l} days to delivery at node {e[1]}' if l > 0
                                else f'Demand from node {e[1]}' for l in range(L)}
                    self.obs_dict[i].update(_new_obs)
                elif i == e[1]:
                    # Upstream connection pipeline
                    L = self.graph.edges[e]['L'] if 'L' in self.graph.edges[e] else 0
                    _obs += L
                    last_entry = max(list(self.obs_dict[i].keys())) + 1
                    _new_obs = {last_entry + l: f'{l} days to receive delivery from node {e[0]}'
                               for l in range(L)}
                    self.obs_dict[i].update(_new_obs)                    
                    
                    _act += 1
                    
            obs_space = spaces.Box(low=0, high=1E6, shape=(_obs,))
            act_space = spaces.Box(low=0, high=1E6, shape=(_act,))
            self.tot_obs_space[i] = obs_space
            self.tot_act_space[i] = act_space
            
#             if self.algo_dict[i][0].upper() == 'RL':
#                 observation_space[i] = obs_space
#                 action_space[i] = act_space
                
        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Dict(action_space)
        
    def _update_state(self):
        states = {}
        for i in self.agent_ids:
            try:
                _state = np.array([self.X.iloc[self.period][i]])
            except KeyError:
                # No inventory on hand
                _state = np.array([0])
            # Get retail demand
            if i in self.retail_supply_nodes:
                demand = np.hstack([self.D[d].iloc[self.period] 
                    for d in self.retail_links if i==d[0]])
                _state = np.hstack([_state, demand])
                
            for e in self.graph.edges():
                # Get demand info from downstream connections
                if i == e[0] and i not in self.retail_supply_nodes:
                    demand = np.array([self.R[e].iloc[self.period]])
                    _state = np.hstack([_state, demand])
                    
                # Get pipeline info from downstream connections
                if i == e[0] and 'L' in self.graph.edges[e]:
                    L = self.graph.edges[e]['L']
                    if L == 0:
                        continue
                    if self.period == 0:
                        _ds_pipe = np.array([self.Y[e].iloc[self.period]])
                    else:
                        _ds_pipe = self.Y[e].iloc[max(self.period-L, 0):self.period].values
                        
                    pl = len(_ds_pipe)
                    if pl < L:
                        ds_pipe = np.zeros(L)
                        ds_pipe[-pl:] += _ds_pipe
                    else:
                        ds_pipe = _ds_pipe.copy()
                        
                    _state = np.hstack([_state, ds_pipe])
                    
                # Get pipeline info from upstream connections
                if i == e[1] and 'L' in self.graph.edges[e]:
                    L = self.graph.edges[e]['L']
                    if L == 0:
                        continue
                    if self.period == 0:
                        _us_pipe = np.array([self.Y[e].iloc[self.period]])
                    else:
                        _us_pipe = self.Y[e].iloc[max(self.period-L, 0):self.period].values
                        
                    pl = len(_us_pipe)
                    if pl < L:
                        us_pipe = np.zeros(L)
                        us_pipe[-pl:] += _us_pipe
                    else:
                        us_pipe = _us_pipe.copy()
                        
                    _state = np.hstack([_state, us_pipe])
            
            states[i] = _state.copy()

        self.state = states
    
    def _Step(self, action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = number of units to request from each supplier.
        dictionary: keys are (supplier, purchaser) tuples
                    values are number of units requested from supplier
                    dimension = len(reorder_links)
        '''
        t = self.period
        if type(action) != dict: #convert to dict if a list was given
            action = {key: action[i] for i, key in enumerate(self.reorder_links)}
        
        #Place Orders
        for key in action.keys():
            request = round(max(action[key],0)) #force to integer value
            supplier = key[0]
            purchaser = key[1]
            if supplier in self.rawmat:
                self.R.loc[t,(supplier, purchaser)] = request #accept request since supply is unlimited
                self.S.loc[t,(supplier, purchaser)] = request
            elif supplier in self.distrib:
                X_supplier = self.X.loc[t,supplier] #request limited by available inventory at beginning of period
                self.R.loc[t,(supplier, purchaser)] = min(request, X_supplier)
                self.S.loc[t,(supplier, purchaser)] = min(request, X_supplier)
            elif supplier in self.factory:
                C = self.graph.nodes[supplier]['C'] #supplier capacity
                v = self.graph.nodes[supplier]['v'] #supplier yield
                X_supplier = self.X.loc[t,supplier] #on-hand inventory at beginning of period
                self.R.loc[t,(supplier, purchaser)] = min(request, C, v*X_supplier)
                self.S.loc[t,(supplier, purchaser)] = min(request, C, v*X_supplier)
            
        #Receive deliveries and update inventories
        for j in self.main_nodes:
            #update pipeline inventories
            incoming = []
            for k in self.graph.predecessors(j):
                L = self.graph.edges[(k,j)]['L'] #extract lead time
                if t - L >= 0: #check if delivery has arrived
                    delivery = self.R.loc[t-L,(k,j)]
                else:
                    delivery = 0
                incoming += [delivery] #update incoming material
                self.Y.loc[t+1,(k,j)] = self.Y.loc[t,(k,j)] - delivery + self.R.loc[t,(k,j)]

            #update on-hand inventory
            if 'v' in self.graph.nodes[j]: #extract production yield
                v = self.graph.nodes[j]['v']
            else:
                v = 1
            outgoing = 1/v * np.sum([self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #consumed inventory (for requests placed)
            self.X.loc[t+1,j] = self.X.loc[t,j] + np.sum(incoming) - outgoing
            
        # demand is realized
        for j in self.retail:
            for k in self.market:
                #read user specified demand. if all zeros, use demand_dist instead.
                Demand = self.graph.edges[(j,k)]['user_D']
                if np.sum(Demand) > 0:
                    self.D.loc[t,(j,k)] = Demand[t]
                else:
                    Demand = self.graph.edges[(j,k)]['demand_dist']
                    self.D.loc[t,(j,k)] = Demand.rvs(**self.graph.edges[(j,k)]['dist_param'])
                if self.backlog and t >= 1:
                    D = self.D.loc[t,(j,k)] + self.U.loc[t-1,(j,k)]
                else:
                    D = self.D.loc[t,(j,k)]
                # Satisfy demand up to available level
                X_retail = self.X.loc[t+1,j] #get inventory at retail before demand was realized
                self.S.loc[t,(j,k)] = min(D, X_retail) #perform sale
                self.X.loc[t+1,j] -= self.S.loc[t,(j,k)] #update inventory
                self.U.loc[t,(j,k)] = D - self.S.loc[t,(j,k)] #update unfulfilled orders

        # calculate profit
        for j in self.main_nodes:
            a = self.alpha
            # Sales revenue
            SR = np.sum([self.graph.edges[(j,k)]['p'] * self.S.loc[t,(j,k)] 
                         for k in self.graph.successors(j)])
            # Purchasing costs
            PC = np.sum([self.graph.edges[(k,j)]['p'] * self.R.loc[t,(k,j)] 
                         for k in self.graph.predecessors(j)])
            if j not in self.rawmat:
                # Holding costs
                HC = self.graph.nodes[j]['h'] * self.X.loc[t+1,j] + \
                    np.sum([self.graph.edges[(k,j)]['g'] * self.Y.loc[t+1,(k,j)] 
                            for k in self.graph.predecessors(j)])
            else:
                HC = 0
            if j in self.factory:
                # Operating costs
                OC = self.graph.nodes[j]['o'] / self.graph.nodes[j]['v'] * \
                    np.sum([self.S.loc[t,(j,k)] 
                            for k in self.graph.successors(j)])
            else:
                OC = 0
            if j in self.retail:
                # Unfulfilled penalty
                UP = np.sum([self.graph.edges[(j,k)]['b'] * self.U.loc[t,(j,k)] 
                             for k in self.graph.successors(j)])
            else:
                UP = 0
            self.P.loc[t,j] = a**t * (SR - PC - OC - HC - UP)
        
        # update period
        self.period += 1

        # set reward (profit from current timestep)
        reward = self.P.loc[t,:].sum()
        
        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False
            # update stae
            self._update_state()

        return self.state, reward, done, {}
    
    def step(self, action):
        self.state, reward, done, info = self._Step(action)
        # TODO: change reward to return reward for each node
        rewards = {i: reward for i in self.agent_ids}
        dones = {i: done for i in self.agent_ids}
        dones['__all__'] = done
        infos = {i: info for i in self.agent_ids}
        
        return self.state, rewards, dones, infos