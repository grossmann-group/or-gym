import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
# path = "c:/Users/HD/Documents/Development/or-gym/scenarios/"
path = "./scenarios/"
ver = "v3"
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
# 986.656
# 1000.029
# 919.2359999999999
# 1009.756
# 941.587
# 906.818
# 960.1500000000001
# 862.1759999999999
# 967.5939999999999
# 819.731
# 960.336
# 1038.2469999999998
# 907.627
# 975.8159999999999
# 869.299
# 954.37
# 1024.1450000000002
# 960.61
# 894.9869999999999
# 896.394
# 942.9000000000001
# 994.434
# 894.848
# 1014.3050000000001
# 931.896
# 1005.9129999999999
# 1045.173
# 844.116
# 966.5339999999999
# 999.0319999999999
# 895.9200000000001
# 990.2739999999999
# 1005.4870000000001
# 917.0959999999999
# 971.2520000000001
# 960.925
# 993.902
# 898.6989999999998
# 870.3749999999998
# 923.9169999999999
# 875.3129999999999
# 885.5429999999999
# 954.9460000000001
# 1000.6359999999999
# 966.1030000000001
# 910.921
# 890.5889999999999
# 947.106
# 891.0109999999997
# 1009.961
# 923.785
# 799.789
# 832.6499999999999
# 996.9479999999999
# 927.9250000000001
# 973.7049999999999
# 1004.589
# 817.9689999999999
# 998.11
# 935.6119999999999
# 965.738
# 983.9960000000001
# 942.968
# 901.924
# 949.106
# 824.833
# 902.857
# 808.906
# 924.865
# 911.058
# 996.5650000000002
# 989.709
# 988.48
# 925.9609999999998
# 979.953
# 963.599
# 1044.4730000000002
# 844.537
# 837.96
# 938.2959999999999
# 942.9330000000002
# 940.4060000000002
# 988.422
# 1030.488
# 903.701
# 872.3009999999999
# 907.778
# 927.9629999999999
# 993.905
# 901.5369999999999
# 996.404
# 930.2289999999999
# 953.5390000000001
# 949.0909999999999
# 920.6280000000002
# 1040.121
# 934.809
# 873.5889999999999
# 997.602
# 825.1289999999999