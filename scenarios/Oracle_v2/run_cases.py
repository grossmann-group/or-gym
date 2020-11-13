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

#solve shrinking horizon model
for i in range(100):
    #create environmnet with user specified demand
    user_D = {(1,0): D[:,i]} #assign scenario to retail/market link
    sample_path = {(1,0): True} #note that the specified demand is sampled from the prob distribution
    env = or_gym.make("InvManagement-"+ver, env_config={'user_D': user_D,
                                                        'sample_path': sample_path})
    #create model
    m=net_im_lp_model(env,perfect_information=True,use_expectation=True)
    #select solver
    s=SolverFactory(solver)
    res=s.solve(m, tee=False)
    if res['Solver'][0]['Termination condition'][:] != 'optimal':
        print("ERROR: NOT OPTIMAL")
    else:
        print(np.sum(list(m.P.get_values().values())))
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

# Cumulative Profit:
# 811.2329999999998
# 987.4680000000001
# 794.4110000000001
# 937.8860000000002
# 972.068
# 785.5419999999999
# 897.8539999999999
# 942.075
# 851.2950000000001
# 901.7930000000001
# 903.79
# 856.7080000000001
# 871.026
# 878.978
# 863.0179999999999
# 804.7739999999999
# 807.8380000000001
# 811.0
# 775.576
# 760.391
# 814.071
# 954.317
# 917.662
# 885.587
# 834.5029999999999
# 845.431
# 802.129
# 950.113
# 879.7699999999999
# 766.9390000000001
# 726.7080000000001
# ERROR: NOT OPTIMAL
# 864.6269999999998
# 937.7090000000001
# 872.6220000000001
# 762.031
# 922.9110000000001
# 865.8979999999999
# 906.0310000000002
# 912.1700000000001
# 854.0710000000004
# 827.8280000000002
# 879.7140000000002
# 762.9570000000001
# 855.7950000000001
# 691.307
# 887.7559999999999
# 854.9269999999999
# 947.7859999999998
# 935.354
# 918.6539999999997
# 869.3889999999999
# 911.253
# 888.495
# 946.267
# 778.455
# 792.7350000000001
# 871.996
# 856.8990000000001
# 839.891
# 939.6109999999999
# 928.2190000000002
# 869.1339999999999
# 786.9770000000001
# 768.329
# 843.384
# 827.1809999999999
# 906.3159999999998
# 885.8109999999997
# 887.8659999999999
# 831.1809999999998
# 829.558
# 943.3039999999999
# 833.2470000000001
# 857.2660000000001
# 877.2070000000001
# 789.7920000000001
# 831.8939999999998
# 812.421
# 912.5749999999998
# 862.9150000000002
# 846.012
# 857.991
# 873.9780000000001
# 956.4749999999999
# 797.4759999999999
# 814.296
# 798.0049999999999
# 896.6179999999999
# 877.7420000000001
# 902.5229999999998
# 989.9269999999999
# 945.1400000000001
# 838.3929999999999
# 848.758
# 829.913
# 851.2430000000002
# 822.2099999999998
# 881.0699999999998