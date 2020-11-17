import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
# path = "c:/Users/HD/Documents/Development/or-gym/scenarios/"
path = "./scenarios/"
ver = "v2"
model = "Oracle"
solver = "glpk"

#create file names
filename = path+"in"+ver+".pkl"
D = pickle.load(open(filename,'rb'))
filesave = path+model+"_"+ver+"/"
f = open(filesave+"results.txt", "w")

#solve shrinking horizon model
for i in range(100):
    #create environmnet with user specified demand
    user_D = {(1,0): D[:,i]} #assign scenario to retail/market link
    sample_path = {(1,0): True} #note that the specified demand is sampled from the prob distribution
    env = or_gym.make("InvManagement-"+ver, env_config={'user_D': user_D,
                                                        'sample_path': sample_path})
    #create model
    m=net_im_lp_model(env,perfect_information=True)
    #select solver
    s=SolverFactory(solver)
    res=s.solve(m, tee=False)
    if res['Solver'][0]['Termination condition'][:] != 'optimal':
        print("Case " + str(i) + " - ERROR: NOT OPTIMAL", file = f)
    else:
        cpu_time = res['Solver'][0]['Time']
        obj_value = np.sum(list(m.P.get_values().values()))
        print_msg = "Case " + str(i) + " - Profit: $" + str(obj_value) + "; CPU = " + str(cpu_time)
        print(print_msg, file = f)
        #extract and pickle data
        X = m.X.get_values()
        for key in X.keys():
            env.X[key[1]][key[0]] = X[key]
        Y = m.Y.get_values()
        for key in Y.keys():
            env.Y[key[1:]][key[0]] = Y[key]
        R = m.R.get_values()
        for key in R.keys():
            env.R[key[1:]][key[0]] = R[key]
        S = m.S.get_values()
        for key in S.keys():
            env.S[key[1:]][key[0]] = S[key]
        U = m.U.get_values()
        for key in U.keys():
            env.U[key[1:]][key[0]] = U[key]
        P = m.P.get_values()
        for key in P.keys():
            env.P[key[1]][key[0]] = P[key]
        env.D[(1,0)] = [d for d in m.D.values()]
        pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

f.close()