import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "./scenarios/"
ver = "v2"
model = "DLP"
mode = "SH"
window = 30
solver = "glpk"

#create file names
filename = path+"in"+ver+".pkl"
D = pickle.load(open(filename,'rb'))
filesave = path+model+"_"+mode+"_"+ver+"/"

#solve shrinking horizon model
for i in range(100):
    #create environmnet with user specified demand
    user_D = {(1,0): D[:,i]} #assign scenario to retail/market link
    sample_path = {(1,0): True} #note that the specified demand is sampled from the prob distribution
    env = or_gym.make("InvManagement-"+ver, env_config={'user_D': user_D,
                                                        'sample_path': sample_path})
    #loop through each period in the simulation, optimizing along the way
    for t in range(env.num_periods):
        #create model
        m=net_im_lp_model(env,window_size=window,use_expectation=True)
        #select solver
        s=SolverFactory(solver)
        #solve model
        res=s.solve(m, tee=False)
        #check result is optimal
        if res['Solver'][0]['Termination condition'][:] != 'optimal':
            print("ERROR: NOT OPTIMAL")
            break
        #extract reorder quantities
        Ropt=m.R.get_values()
        #pass action for the current timestep only (i.e. t=0)
        action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
        outcome=env.step(action)
    #print profit for the total simulation run
    print("Case " + str(i) + " - " + str(env.P.sum().sum()))
    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

# Cumulative Profits:
# Case 0 - 767.472
# Case 1 - 902.6880000000001
# Case 2 - 804.315
# Case 3 - 854.0160000000001
# Case 4 - 859.086
# Case 5 - 824.8399999999999
# Case 6 - 857.886
# Case 7 - 841.5989999999999
# Case 8 - 782.482
# Case 9 - 842.7719999999999
# Case 10 - 861.606
# Case 11 - 793.9260000000002
# Case 12 - 853.2360000000001
# Case 13 - 847.98
# Case 14 - 852.0160000000001
# Case 15 - 785.8220000000001
# Case 16 - 814.9360000000001
# Case 17 - 798.935
# Case 18 - 827.2859999999998
# Case 19 - 794.171
# Case 20 - 885.75
# Case 21 - 873.152
# Case 22 - 845.5509999999999
# Case 23 - 870.4390000000001
# Case 24 - 838.8389999999999
# Case 25 - 823.0610000000001
# Case 26 - 768.0480000000002
# Case 27 - 860.135
# Case 28 - 832.598
# Case 29 - 761.7920000000001
# Case 30 - 732.9990000000003
# Case 31 - 849.8599999999999
# Case 32 - 834.1680000000001
# Case 33 - 785.5120000000002
# Case 34 - 851.472
# Case 35 - 761.8120000000001
# Case 36 - 840.9250000000002
# Case 37 - 830.76
# Case 38 - 842.8509999999999
# Case 39 - 844.1100000000001
# Case 40 - 835.0889999999999
# Case 41 - 776.4780000000001
# Case 42 - 820.421
# Case 43 - 728.0729999999999
# Case 44 - 786.454
# Case 45 - 759.7570000000001
# Case 46 - 855.989
# Case 47 - 842.3520000000001
# Case 48 - 869.825
# Case 49 - 847.2250000000001
# Case 50 - 832.74
# Case 51 - 848.6969999999999
# Case 52 - 911.2819999999999
# Case 53 - 833.306
# Case 54 - 838.1010000000001
# Case 55 - 781.2520000000002
# Case 56 - 801.8880000000001
# Case 57 - 793.367
# Case 58 - 845.1510000000003
# Case 59 - 803.0619999999999
# Case 60 - 857.7310000000002
# Case 61 - 862.876
# Case 62 - 815.54
# Case 63 - 840.1860000000001
# Case 64 - 750.0720000000001
# Case 65 - 797.1399999999999
# Case 66 - 812.703
# Case 67 - 844.8230000000001
# Case 68 - 880.039
# Case 69 - 815.5909999999999
# Case 70 - 817.6570000000002
# Case 71 - 748.762
# Case 72 - 827.7429999999999
# Case 73 - 859.8779999999999
# Case 74 - 838.6980000000001
# Case 75 - 850.6070000000002
# Case 76 - 866.692
# Case 77 - 811.1239999999998
# Case 78 - 794.3320000000001
# Case 79 - 789.0690000000002
# Case 80 - 805.9580000000001
# Case 81 - 818.9090000000001
# Case 82 - 835.829
# Case 83 - 800.0029999999999
# Case 84 - 816.521
# Case 85 - 874.163
# Case 86 - 846.1049999999998
# Case 87 - 862.8380000000002
# Case 88 - 777.5250000000001
# Case 89 - 819.4320000000002
# Case 90 - 831.5439999999999
# Case 91 - 833.7629999999999
# Case 92 - 850.8490000000002
# Case 93 - 854.415
# Case 94 - 798.4720000000002
# Case 95 - 823.6989999999998
# Case 96 - 866.8779999999999
# Case 97 - 796.9650000000001
# Case 98 - 752.9560000000001
# Case 99 - 871.815