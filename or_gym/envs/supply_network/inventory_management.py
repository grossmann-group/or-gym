'''
Multi-period inventory management
Hector Perez, Christian Hubbs, Can Li
9/14/2020
'''

import gym
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *
from or_gym.utils import assign_env_config
from collections import deque

class NetInvMgmtMasterEnv(gym.Env):
    '''
    The supply network environment is structured as follows:
    
    It is a multi-period multi-node production-inventory system for 
    a single non-perishable product that is sold in discrete quantities. 
    Two main types of nodes exist: 1) production nodes (which have an 
    inventory holding area and a manufacturing area), and 2) distribution
    nodes (which only have an inventory holding area). Retail nodes are
    considered distribution nodes. Other node types in the network include 
    1) raw material nodes (source nodes), which have unlimited supply
    of raw materials, and 2) market nodes (sink nodes), which generate an 
    uncertain demand on their respective retailers in each period. 

    Within production nodes, the inventory holding area holds the inventory 
    necessary to produce the respective intermediate material at that node. 
    Yield ratios are specified at each production stage relating the amount 
    of material produced from one unit of inventory. Production at each node
    is bounded by the nodes's production capacity and the available inventory.

    Lead times between neighbor nodes exist and are associated with the edges
    connecting them.
        
    At the beginning of each time period, the following sequence of events occurs:
    
    0) Each node places replenishment orders to their immediate suppliers. 
       Replenishment orders are filled according to available production capacity 
       and available inventory at the suppliers. There is a cost associated with
       each order request.
    1) Incoming inventory replenishment shipments that have made it down network
       pipeline (after the associated lead time) are received at each node.
    2) Market demands occur at the retail nodes. Demands are sampled from a 
       specified discrete probability distribution. Demands are filled according 
       to available inventory at the retailers.
    4) Option: one of the following occurs,
        a) Unfulfilled sales are backlogged at a penalty. 
            Note: Backlogged orders are added to the next period's market demand.
        b) Unfulfilled sales and replenishment orders are lost with a 
           goodwill-loss penalty. 
    5) Surpluss inventory is held at each stage at a holding cost. 
        Pipeline inventories also incur a cost for each period in the pipeline.
        
    '''
    def __init__(self, *args, **kwargs):
        '''
        num_periods = number of periods in simulation.
        Node specific parameters:
            - I0 = initial inventory.
            - C = production capacity.
            - v = production yield in the range (0, 1].
            - o = unit operating cost (feed-based)
            - h = unit holding cost for excess on-hand inventory.
        Edge specific parameters:
            - L = lead times in betwen adjacent nodes.
            - p = unit price to send material between adjacent nodes (purchase price/reorder cost)
            - b = unit backlog cost or good-wil loss for unfulfilled market demand between adjacent retailer and market.
            - g = unit holding cost for pipeline inventory on a specified edge.
            - prob_dist = probability distribution function on a (retailer, market) edge.
            - demand_dist = demand distribution for (retailer, market) edge. Two options:
                - use scipy probability distribution: must be a lambda function calling the rvs method of the distribution
                    i.e. lambda: poisson.rvs(mu=20)
                - use a list of user specified demands for each period. 
        backlog = Are unfulfilled orders backlogged? True = backlogged, False = lost sales.
        prob_dist = Value between 1 and 4. Specifies distribution for customer demand.
            1: poisson distribution
            2: binomial distribution
            3: uniform random integer
            4: geometric distribution
            5: user supplied demand values
        dist_param = named values for parameters fed to statistical distribution.
            poisson: {'mu': <mean value>}
            binom: {'n': <mean value>, 
                    'p': <probability between 0 and 1 of getting the mean value>}
            raindint: {'low' = <lower bound>, 'high': <upper bound>}
            geom: {'p': <probability. Outcome is the number of trials to success>}
        alpha = discount factor in the range (0,1] that accounts for the time value of money
        seed_int = integer seed for random state.
        user_D = dataframe containing user specified demand for each (retail, market) pair at
            each time period in the simulation.
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self._max_rewards = 2000
        self.num_periods = 30
        self.backlog = True
        self.alpha = 0.97
        self.seed_int = 0
        self.user_D = pd.DataFrame(data = np.zeros([self.num_periods, 1]), 
                                   columns = pd.MultiIndex.from_tuples([(1,0)], names = ['Retailer','Market']))
        self._max_rewards = 2000
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([0])
        self.graph.add_nodes_from([1], I0 = 100,
                                        h = 0.150)
        self.graph.add_nodes_from([2], I0 = 100,
                                        C = 100,
                                        o = 0.000,
                                        v = 1.000,
                                        h = 0.100)
        self.graph.add_nodes_from([3], I0 = 200,
                                        C = 90,
                                        o = 0.000,
                                        v = 1.000,
                                        h = 0.050)
        self.graph.add_nodes_from([4], I0 = 1000,
                                        C = 80,
                                        o = 0.500,
                                        v = 1.000,
                                        h = 0.000)
        self.graph.add_nodes_from([5])
        self.graph.add_edges_from([(1,0,{'p': 2.00,
                                         'b': 0.10,
                                         'demand_dist': lambda: poisson.rvs(mu=20)}),
                                   (2,1,{'L': 3,
                                         'p': 1.50,
                                         'g': 0.00}),
                                   (3,2,{'L': 5,
                                         'p': 1.00,
                                         'g': 0.00}),
                                   (4,3,{'L': 10,
                                         'p': 0.75,
                                         'g': 0.00}),
                                   (5,4,{'L': 0,
                                         'p': 0.00,
                                         'g': 0.00})])
        
        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)
        
        # # input parameters
        # try:
        #     self.init_inv = np.array(list(self.I0))
        # except:
        #     self.init_inv = np.array([self.I0])
        # self.num_periods = self.periods
        # self.unit_price = np.append(self.p,self.r[:-1]) # cost to stage 1 is price to stage 2
        # self.unit_cost = np.array(self.r)
        # self.demand_cost = np.array(self.k)
        # self.holding_cost = np.append(self.h,0) # holding cost at last stage is 0
        # try:
        #     self.supply_capacity = np.array(list(self.c))
        # except:
        #     self.supply_capacity = np.array([self.c])
        # try:
        #     self.lead_time = np.array(list(self.L))
        # except:
        #     self.lead_time = np.array([self.L])
        # self.discount = self.alpha
        # self.user_D = np.array(list(self.user_D))
        # self.num_stages = len(self.init_inv) + 1
        # m = self.num_stages
        # lt_max = self.lead_time.max()
        
        #  parameters
        self.num_nodes = self.graph.number_of_nodes()
        self.markets = [j for j in self.graph.nodes() if len(list(self.graph.successors(j))) == 0]
        self.distrib = [j for j in self.graph.nodes() if 'C' not in self.graph.nodes[j] and 'I0' in self.graph.nodes[j]]
        self.retail = [j for j in self.graph.nodes() if len(set.intersection(set(self.graph.successors(j)), set(self.markets))) > 0]
        self.factory = [j for j in self.graph.nodes() if 'C' in self.graph.nodes[j]]
        self.rawmat = [j for j in self.graph.nodes() if len(list(self.graph.predecessors(j))) == 0]
        self.main_nodes = np.sort(self.distrib + self.factory)
        self.reorder_links = [e for e in self.graph.edges() if 'L' in self.graph.edges[e]] #exclude links to markets (these cannot have lead time 'L')
        self.retail_links = [e for e in self.graph.edges() if 'L' not in self.graph.edges[e]] #links joining retailers to markets
        self.sales_links = set.union(set(self.reorder_links), set(self.retail_links)) #all links involved in sale in the network

        # check inputs
        assert set(self.graph.nodes()) == set.union(set(self.markets),
                                                    set(self.distrib),
                                                    set(self.factory),
                                                    set(self.rawmat)), "The union of market, distribution, factory, and raw material nodes is not equal to the system nodes."
        for j in self.graph.nodes():
            if 'I0' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['I0'] >= 0, "The initial inventory cannot be negative for node {}.".format(j)
            if 'h' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['h'] >= 0, "The inventory holding costs cannot be negative for node {}.".format(j)
            if 'C' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['C'] > 0, "The production capacity must be positive for node {}.".format(j)
            if 'o' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['o'] >= 0, "The operating costs cannot be negative for node {}.".format(j)
            if 'v' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['v'] > 0 and self.graph.nodes[j]['v'] <= 1, "The production yield must be in the range (0, 1] for node {}.".format(j)
        for e in self.graph.edges():
            if 'L' in self.graph.edges[e]:
                assert self.graph.edges[e]['L'] >= 0, "The lead time joining nodes {} cannot be negative.".format(e)
            if 'p' in self.graph.edges[e]:
                assert self.graph.edges[e]['p'] >= 0, "The sales price joining nodes {} cannot be negative.".format(e)
            if 'b' in self.graph.edges[e]:
                assert self.graph.edges[e]['b'] >= 0, "The unfulfilled demand costs joining nodes {} cannot be negative.".format(e)
            if 'g' in self.graph.edges[e]:
                assert self.graph.edges[e]['g'] >= 0, "The pipeline inventory holding costs joining nodes {} cannot be negative.".format(e)
            if 'demand_dist' in self.graph.edges[e]:
                if isinstance(self.graph.edges[e]['demand_dist'], list):
                    assert len(self.graph.edges[e]['demand_dist']) == self.num_periods, "The user specified demand joining (retailer, market): {} must be of length {}.".format(e,self.num_periods)
        assert self.backlog == False or self.backlog == True, "The backlog parameter must be a boolean."
        assert self.graph.number_of_nodes() >= 2, "The minimum number of nodes is 2. Please try again"
        assert self.alpha>0 and self.alpha<=1, "alpha must be in the range (0, 1]."
        
        # # select distribution
        # self.demand_dist = distributions[self.dist]  
        
        # set random generation seed (unless using user demands)
        # if self.dist < 5:
        self.seed(self.seed_int)
        
        # action space (reorder quantities for each node for each supplier; list)
        # An action is defined for every node
        num_reorder_links = len(self.reorder_links) 
        self.lt_max = np.max([self.graph.edges[e]['L'] for e in self.graph.edges() if 'L' in self.graph.edges[e]])
        self.init_inv_max = np.max([self.graph.nodes[j]['I0'] for j in self.graph.nodes() if 'I0' in self.graph.nodes[j]])
        self.capacity_max = np.max([self.graph.nodes[j]['C'] for j in self.graph.nodes() if 'C' in self.graph.nodes[j]])
        self.pipeline_length = len(self.main_nodes)*(self.lt_max+1)
        self.action_space = gym.spaces.Box(
            low=np.zeros(num_reorder_links), 
            high=np.ones(num_reorder_links)*(self.init_inv_max + self.capacity_max*self.num_periods), 
            dtype=np.int32)
        # observation space (total inventory at each node, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(self.pipeline_length)*(self.init_inv_max + self.capacity_max*self.num_periods)*10,
            high=np.ones(self.pipeline_length)*(self.init_inv_max + self.capacity_max*self.num_periods), 
            dtype=np.int32)

        # intialize
        self.reset()

    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))
        
    def reset(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        '''
        T = self.num_periods
        J = len(self.main_nodes)
        RM = len(self.retail_links) #number of retailer-market pairs
        PS = len(self.reorder_links) #number of purchaser-supplier pairs in the network
        SL = len(self.sales_links) #number of edges in the network (excluding links form raw material nodes)
        
        # simulation result lists
        self.X=pd.DataFrame(data = np.zeros([T + 1, J]), columns = self.main_nodes) # inventory at the beginning of each period
        self.Y=pd.DataFrame(data = np.zeros([T + 1, PS]), columns = pd.MultiIndex.from_tuples(self.reorder_links, names = ['Source','Receiver'])) # pipeline inventory at the beginning of each period
        self.R=pd.DataFrame(data = np.zeros([T, PS]), columns = pd.MultiIndex.from_tuples(self.reorder_links, names = ['Supplier','Requester'])) # replenishment orders
        self.S=pd.DataFrame(data = np.zeros([T, SL]), columns = pd.MultiIndex.from_tuples(self.sales_links, names = ['Seller','Purchaser'])) # units sold
        self.D=pd.DataFrame(data = np.zeros([T, RM]), columns = pd.MultiIndex.from_tuples(self.retail_links, names = ['Retailer','Market'])) # demand at retailers
        self.U=pd.DataFrame(data = np.zeros([T, RM]), columns = pd.MultiIndex.from_tuples(self.retail_links, names = ['Retailer','Market'])) # unfulfilled demand for each market - retailer pair
        self.P=pd.DataFrame(data = np.zeros([T, J]), columns = self.main_nodes) # profit at each node
        
        # initializetion
        self.period = 0 # initialize time
        for j in self.main_nodes:
            self.X.loc[0,j]=self.graph.nodes[j]['I0'] # initial inventory
        self.Y.loc[0,:]=np.zeros(PS) # initial pipeline inventory 
        self.action_log = np.zeros([T, PS])

        # set state
        self._update_state()
        
        return self.state

    def _update_state(self):
        m = len(self.main_nodes)
        t = self.period
        state = np.zeros(self.pipeline_length)
        state[:m] = self.X.loc[t,:]
        if t == 0:
            pass
        elif t >= self.lt_max:
            state[-m*self.lt_max:] += self.action_log[t-self.lt_max:t].flatten()
        else:
            state[-m*t:] += self.action_log[:t].flatten()

        self.state = state.copy()
    
    # def _update_base_stock_policy_state(self):
    #     '''
    #     Get current state of the system: Inventory position at each echelon
    #     Inventory at hand + Pipeline inventory - backlog up to the current stage 
    #     (excludes last stage since no inventory there, nor replenishment orders placed there).
    #     '''
    #     n = self.period
    #     m = self.num_stages
    #     if n>=1:
    #         IP = np.cumsum(self.I[n,:] + self.T[n,:] - self.B[n-1,:-1])
    #     else:
    #         IP = np.cumsum(self.I[n,:] + self.T[n,:])
    #     self.state = IP
    
    def step(self,action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = number of units to request from each supplier.
            dictionary: keys are (supplier, purchaser) tuples
                        values are number of units requested from supplier
                        dimension = len(reorder_links) (number of edges joining all nodes, 
                                                        except market nodes)
        '''
        t = self.period
        
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
            for k in self.markets:
                Demand = self.graph.edges[(j,k)]['demand_dist']
                if isinstance(Demand, list):
                    D = Demand[t]
                else:
                    D = Demand()
                if self.backlog and t >= 1:
                    self.D.loc[t,(j,k)] = D + self.U.loc[t-1,(j,k)]
                else:
                    self.D.loc[t,(j,k)] = D
                #satisfy demand up to available level
                X_retail = self.X.loc[t+1,j] #get inventory at retail before demand was realized
                self.S.loc[t,(j,k)] = min(self.D.loc[t,(j,k)], X_retail) #perform sale
                self.X.loc[t+1,j] -= self.S.loc[t,(j,k)] #update inventory
                self.U.loc[t,(j,k)] = self.D.loc[t,(j,k)] - self.S.loc[t,(j,k)] #update unfulfilled orders

        # calculate profit
        for j in self.main_nodes:
            a = self.alpha
            SR = np.sum([self.graph.edges[(j,k)]['p'] * self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #sales revenue
            PC = np.sum([self.graph.edges[(k,j)]['p'] * self.R.loc[t,(k,j)] for k in self.graph.predecessors(j)]) #purchasing costs
            if j not in self.rawmat:
                HC = self.graph.nodes[j]['h'] * self.X.loc[t+1,j] + np.sum([self.graph.edges[(k,j)]['g'] * self.Y.loc[t+1,(k,j)] for k in self.graph.predecessors(j)]) #holding costs
            else:
                HC = 0
            if j in self.factory:
                OC = self.graph.nodes[j]['o'] / self.graph.nodes[j]['v'] * np.sum([self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #operating costs
            else:
                OC = 0
            if j in self.retail:
                UP = np.sum([self.graph.edges[(j,k)]['b'] * self.U.loc[t,(j,k)] for k in self.graph.successors(j)]) #unfulfilled penalty
            else:
                UP = 0
            self.P.loc[t,j] = a**t * (SR - PC - OC - HC - UP)
        
        # update period
        self.period += 1  
        
        # update stae
        self._update_state()
        
        # set reward (profit from current timestep)
        reward = self.P.loc[t,:].sum()
        
        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False
            
        return self.state, reward, done, {}
    
    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        return self.action_space.sample()
        
    # def base_stock_action(self,z):
    #     '''
    #     Sample action (number of units to request) based on a base-stock policy (order up to z policy)
    #     z = [integer list; dimension |Stages| - 1] base stock level (no inventory at the last stage)
    #     '''
    #     n = self.period
    #     c = self.supply_capacity
    #     m = self.num_stages
    #     IP = self._update_base_stock_policy_state() # extract inventory position (current state)
        
    #     try:
    #         dimz = len(z)
    #     except:
    #         dimz = 1
    #     assert dimz == m-1, "Wrong dimension on base stock level vector. Should be # Stages - 1."
        
    #     # calculate total inventory position at the beginning of period n
    #     R = z - IP # replenishmet order to reach zopt

    #     # check if R can actually be fulfilled (capacity and inventory constraints)
    #     Im1 = np.append(self.I[n,1:], np.Inf) # available inventory at the m+1 stage
    #                                         # NOTE: last stage has unlimited raw materials
    #     Rpos = np.column_stack((np.zeros(len(R)),R)) # augmented materix to get replenishment only if positive
    #     A = np.column_stack((c, np.max(Rpos,axis=1), Im1)) # augmented matrix with c, R, and I_m+1 as columns
        
    #     R = np.min(A, axis = 1) # replenishmet order to reach zopt (capacity constrained)
        
    #     return R
        
class NetInvMgmtBacklogEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class NetInvMgmtLostSalesEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backlog = False
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.pipeline_length), # Never goes negative without backlog
            high=np.ones(self.pipeline_length)*(self.init_inv_max + self.capacity_max*self.num_periods), 
            dtype=np.int32)