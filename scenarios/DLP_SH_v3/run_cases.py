import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "./scenarios/"
ver = "v3"
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
# Case 0 - 787.7260000000001
# Case 1 - 785.5729999999999
# Case 2 - 802.96
# Case 3 - 816.325
# Case 4 - 749.838
# Case 5 - 773.5500000000002
# Case 6 - 818.528
# Case 7 - 795.2939999999999
# Case 8 - 788.7540000000001
# Case 9 - 753.3389999999997
# Case 10 - 796.8599999999999
# Case 11 - 836.768
# Case 12 - 748.1850000000002
# Case 13 - 825.5369999999998
# Case 14 - 740.673
# Case 15 - 801.6140000000003
# Case 16 - 824.7080000000001
# Case 17 - 767.9989999999998
# Case 18 - 754.7070000000001
# Case 19 - 731.8519999999999
# Case 20 - 790.2710000000002
# Case 21 - 792.2240000000004
# Case 22 - 745.5319999999999
# Case 23 - 827.664
# Case 24 - 737.1969999999999
# Case 25 - 803.098
# Case 26 - 813.7430000000002
# Case 27 - 763.701
# Case 28 - 825.8980000000001
# Case 29 - 814.277
# Case 30 - 760.6660000000002
# Case 31 - 803.1120000000001
# Case 32 - 833.8199999999999
# Case 33 - 760.9260000000002
# Case 34 - 820.1679999999999
# Case 35 - 789.145
# Case 36 - 817.239
# Case 37 - 782.624
# Case 38 - 757.0730000000001
# Case 39 - 768.1220000000001
# Case 40 - 793.6859999999997
# Case 41 - 758.8399999999997
# Case 42 - 795.8860000000004
# Case 43 - 821.7550000000001
# Case 44 - 800.4159999999999
# Case 45 - 812.7149999999999
# Case 46 - 778.3770000000002
# Case 47 - 793.867
# Case 48 - 748.982
# Case 49 - 809.673
# Case 50 - 806.7900000000002
# Case 51 - 745.7480000000003
# Case 52 - 711.892
# Case 53 - 797.5490000000002
# Case 54 - 791.623
# Case 55 - 799.9820000000002
# Case 56 - 781.5550000000001
# Case 57 - 760.0150000000001
# Case 58 - 792.3850000000002
# Case 59 - 790.9490000000003
# Case 60 - 784.8120000000001
# Case 61 - 792.2760000000001
# Case 62 - 774.0110000000002
# Case 63 - 758.2870000000003
# Case 64 - 803.117
# Case 65 - 688.1030000000002
# Case 66 - 716.7649999999999
# Case 67 - 746.4279999999999
# Case 68 - 806.096
# Case 69 - 823.4350000000002
# Case 70 - 824.5549999999998
# Case 71 - 819.414
# Case 72 - 825.5930000000001
# Case 73 - 814.4200000000001
# Case 74 - 817.5749999999998
# Case 75 - 769.8200000000002
# Case 76 - 825.4350000000002
# Case 77 - 764.1860000000001
# Case 78 - 758.9869999999999
# Case 79 - 793.7910000000002
# Case 80 - 803.338
# Case 81 - 779.126
# Case 82 - 810.5919999999999
# Case 83 - 830.5050000000001
# Case 84 - 768.4560000000004
# Case 85 - 808.53
# Case 86 - 741.682
# Case 87 - 747.6599999999999
# Case 88 - 786.569
# Case 89 - 761.586
# Case 90 - 818.2730000000001
# Case 91 - 745.6800000000001
# Case 92 - 807.9860000000001
# Case 93 - 753.721
# Case 94 - 804.6400000000003
# Case 95 - 808.1340000000002
# Case 96 - 786.123
# Case 97 - 805.0449999999998
# Case 98 - 842.0450000000001
# Case 99 - 776.3660000000002