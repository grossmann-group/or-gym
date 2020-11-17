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
        print("Case " + str(i) + " - ERROR: NOT OPTIMAL")
    else:
        print("Case " + str(i) + " - " + str(np.sum(list(m.P.get_values().values()))))
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
# Case 0 - 866.2189999999998
# Case 1 - 880.1780000000001
# Case 2 - 837.0769999999999
# Case 3 - 888.208
# Case 4 - 830.4430000000001
# Case 5 - 802.903
# Case 6 - 838.759
# Case 7 - 858.279
# Case 8 - 860.994
# Case 9 - 801.7929999999999
# Case 10 - 841.6459999999998
# Case 11 - 922.7539999999999
# Case 12 - 813.9879999999999
# Case 13 - 857.422
# Case 14 - 797.2830000000001
# Case 15 - 878.8810000000001
# Case 16 - 933.8049999999998
# Case 17 - 846.545
# Case 18 - 814.9369999999999
# Case 19 - 776.0450000000001
# Case 20 - 854.26
# Case 21 - 877.8340000000001
# Case 22 - 805.373
# Case 23 - 893.906
# Case 24 - 812.6850000000001
# Case 25 - 888.408
# Case 26 - 941.1419999999999
# Case 27 - 812.1510000000001
# Case 28 - 930.8679999999999
# Case 29 - 877.3349999999999
# Case 30 - 775.1089999999999
# Case 31 - 871.6089999999999
# Case 32 - 939.311
# Case 33 - 804.3829999999999
# Case 34 - 888.798
# Case 35 - 848.827
# Case 36 - 874.46
# Case 37 - 877.4259999999999
# Case 38 - 813.1889999999999
# Case 39 - 806.0139999999998
# Case 40 - 867.7989999999998
# Case 41 - 817.7640000000001
# Case 42 - 880.5500000000002
# Case 43 - 882.9780000000001
# Case 44 - 869.192
# Case 45 - 902.377
# Case 46 - 833.334
# Case 47 - 825.777
# Case 48 - 780.015
# Case 49 - 888.8900000000001
# Case 50 - 861.675
# Case 51 - 765.5260000000001
# Case 52 - 729.9469999999999
# Case 53 - 875.9850000000001
# Case 54 - 863.227
# Case 55 - 932.8409999999999
# Case 56 - 886.659
# Case 57 - 761.0049999999999
# Case 58 - 875.8649999999999
# Case 59 - 874.3989999999999
# Case 60 - 853.248
# Case 61 - 864.0409999999999
# Case 62 - 827.199
# Case 63 - 778.4050000000002
# Case 64 - 833.45
# Case 65 - 714.9419999999999
# Case 66 - 780.558
# Case 67 - 789.3709999999999
# Case 68 - 855.4689999999998
# Case 69 - 948.4959999999999
# Case 70 - 884.365
# Case 71 - 869.842
# Case 72 - 868.671
# Case 73 - 848.703
# Case 74 - 871.7710000000001
# Case 75 - 854.41
# Case 76 - 923.5009999999999
# Case 77 - 786.762
# Case 78 - 778.144
# Case 79 - 815.0070000000001
# Case 80 - 948.91
# Case 81 - 819.0049999999999
# Case 82 - 868.4649999999999
# Case 83 - 910.936
# Case 84 - 795.9009999999998
# Case 85 - 874.5759999999998
# Case 86 - 788.802
# Case 87 - 810.842
# Case 88 - 873.9290000000001
# Case 89 - 848.555
# Case 90 - 875.376
# Case 91 - 822.9300000000001
# Case 92 - 834.1070000000001
# Case 93 - 829.39
# Case 94 - 854.8220000000001
# Case 95 - 917.0259999999998
# Case 96 - 916.7070000000001
# Case 97 - 871.923
# Case 98 - 949.4190000000001
# Case 99 - 800.5420000000001