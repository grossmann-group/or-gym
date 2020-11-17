import or_gym
import pickle
import numpy as np
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "./scenarios/"
ver = "v2"
model = "DLP"
mode = "RH"
window = 10
solver = "gurobi"

#create file names
filename = path+"in"+ver+".pkl"
D = pickle.load(open(filename,'rb'))
filesave = path+model+"_"+mode+"_"+ver+"/"
f = open(filesave+"results.txt", "w")
CPU_TIMES = np.zeros([100,30])

for i in range(100):
    #create environmnet with user specified demand
    user_D = {(1,0): D[:,i]} #assign scenario to retail/market link
    sample_path = {(1,0): True} #note that the specified demand is sampled from the prob distribution
    env = or_gym.make("InvManagement-"+ver, env_config={'user_D': user_D,
                                                        'sample_path': sample_path})
    #loop through each period in the simulation, optimizing along the way
    cpu_time = 0
    for t in range(env.num_periods):
        #create model
        m=net_im_lp_model(env,window_size=window)
        #select solver
        s=SolverFactory(solver)
        #solve model
        res=s.solve(m, tee=False)
        #check result is optimal
        if str(res['Solver'][0]['Termination condition']) != 'optimal':
            print("ERROR: NOT OPTIMAL")
            break
        #extract reorder quantities
        Ropt=m.R.get_values()
        #pass action for the current timestep only (i.e. t=0)
        action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
        outcome=env.step(action)
        CPU_TIMES[i,t] = res['Solver'][0]['Time']
        cpu_time += CPU_TIMES[i,t]
    #print profit for the total simulation run
    obj_value = env.P.sum().sum()
    print_msg = "Case " + str(i) + " - Profit: $" + str(obj_value) + "; CPU = " + str(cpu_time)
    print(print_msg, file = f)
    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

f.close()

#save CPU times
np.savetxt(filesave+"results_cpu_times.csv", CPU_TIMES, delimiter=",")