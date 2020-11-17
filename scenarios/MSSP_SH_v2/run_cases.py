import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "../../scenarios/"
ver = "v2"
model = "MSSP"
mode = "SH"
window = 10
solver = "gurobi"

#create file names
filename = path+"in"+ver+".pkl"
D = pickle.load(open(filename,'rb'))
filesave = path+model+"_"+mode+"_"+ver+"/"

#solve shrinking horizon model
total_profit_record = []
non_optimal_record = []
cpu_time_record = []
wall_time_record = []
import time

for i in range(100):
    start_time = time.time()
    #create environmnet with user specified demand
    user_D = {(1,0): D[:,i]} #assign scenario to retail/market link
    sample_path = {(1,0): True} #note that the specified demand is sampled from the prob distribution
    env = or_gym.make("InvManagement-"+ver, env_config={'user_D': user_D,
                                                        'sample_path': sample_path})
    #loop through each period in the simulation, optimizing along the way
    for t in range(env.num_periods):
        #create model
        m=net_im_stoch_lp_model(env)
        #select solver
        s=SolverFactory(solver)
        s.options['threads'] = 3
        #solve model
        res=s.solve(m, tee=False)
        #check result is optimal
        if str(res['Solver'][0]['Termination condition']) != 'optimal':
            print("ERROR: NOT OPTIMAL")
            non_optimal_record.append(i)
            break
        #extract reorder quantities
        Ropt=m.R.get_values()
        #pass action for the current timestep only (i.e. t=0)
        action={e[2:]:Ropt[e] for e in Ropt.keys() if (e[0]==0 and e[1]==0)}
        outcome=env.step(action)
        cpu_time_record.append(res['Solver'][0]['Time'])
    #print profit for the total simulation run
    print(env.P.sum().sum())
    total_profit_record.append(env.P.sum().sum())
    end_time = time.time()
    wall_time_record.append(end_time-start_time)

    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))
    # break 
    # if i >=5:
    #     break 
print(filesave)

cpu_time_record_iter = []
for i in range(100):
    cpu_time_record_iter.append(0)
    for j in range(10):
        cpu_time_record_iter[i] += cpu_time_record[i*30+j]

# Cumulative Profits:
