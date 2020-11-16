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
# Case 0 - 718.801
# Case 1 - 864.7269999999999
# Case 2 - 672.6819999999999
# Case 3 - 820.5790000000001
# Case 4 - 863.9749999999999
# Case 5 - 744.839
# Case 6 - 858.8650000000002
# Case 7 - 823.8169999999999
# Case 8 - 726.886
# Case 9 - 782.736
# Case 10 - 846.586
# Case 11 - 744.0409999999999
# Case 12 - 789.394
# Case 13 - 771.662
# Case 14 - 741.8430000000001
# Case 15 - 784.6579999999999
# Case 16 - 737.6609999999998
# Case 17 - 695.9899999999999
# Case 18 - 769.5500000000001
# Case 19 - 712.7439999999998
# Case 20 - 748.171
# Case 21 - 835.098
# Case 22 - 797.1980000000001
# Case 23 - 839.7520000000001
# Case 24 - 770.58
# Case 25 - 723.6099999999999
# Case 26 - 694.7080000000001
# Case 27 - 826.655
# Case 28 - 795.2610000000002
# Case 29 - 702.73
# Case 30 - 656.3590000000002
# Case 31 - 775.498
# Case 32 - 814.6569999999999
# Case 33 - 853.0040000000001
# Case 34 - 758.7140000000002
# Case 35 - 714.952
# Case 36 - 801.847
# Case 37 - 788.642
# Case 38 - 791.173
# Case 39 - 790.9779999999998
# Case 40 - 765.826
# Case 41 - 701.562
# Case 42 - 770.03
# Case 43 - 643.626
# Case 44 - 735.326
# Case 45 - 671.175
# Case 46 - 823.4679999999998
# Case 47 - 906.0090000000001
# Case 48 - 832.2330000000001
# Case 49 - 818.7719999999999
# Case 50 - 799.574
# Case 51 - 792.759
# Case 52 - 828.374
# Case 53 - 779.655
# Case 54 - 826.508
# Case 55 - 702.5780000000001
# Case 56 - 748.096
# Case 57 - 751.813
# Case 58 - 873.4369999999999
# Case 59 - 722.984
# Case 60 - 818.607
# Case 61 - 807.3680000000002
# Case 62 - 759.309
# Case 63 - 791.218
# Case 64 - 645.8209999999999
# Case 65 - 746.9419999999999
# Case 66 - 729.751
# Case 67 - 787.377
# Case 68 - 785.7470000000001
# Case 69 - 767.636
# Case 70 - 767.0709999999999
# Case 71 - 722.1989999999998
# Case 72 - 792.6440000000001
# Case 73 - 821.4649999999999
# Case 74 - 836.8989999999999
# Case 75 - 823.837
# Case 76 - 851.565
# Case 77 - 761.158
# Case 78 - 715.31
# Case 79 - 735.017
# Case 80 - 805.7869999999998
# Case 81 - 746.335
# Case 82 - 742.0640000000003
# Case 83 - 751.4529999999999
# Case 84 - 751.1060000000001
# Case 85 - 836.5039999999999
# Case 86 - 802.6349999999999
# Case 87 - 828.5359999999998
# Case 88 - 744.306
# Case 89 - 776.3419999999999
# Case 90 - 759.3020000000001
# Case 91 - 785.982
# Case 92 - 870.1629999999999
# Case 93 - 875.785
# Case 94 - 723.183
# Case 95 - 727.4530000000001
# Case 96 - 792.385
# Case 97 - 738.327
# Case 98 - 702.956
# Case 99 - 760.0899999999999