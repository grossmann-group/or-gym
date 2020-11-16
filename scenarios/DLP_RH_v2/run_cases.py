import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "./scenarios/"
ver = "v2"
model = "DLP"
mode = "RH"
window = 10
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
# Case 0 - 714.4750000000001
# Case 1 - 869.48
# Case 2 - 754.6949999999999
# Case 3 - 840.2750000000001
# Case 4 - 870.6000000000001
# Case 5 - 765.5849999999998
# Case 6 - 859.395
# Case 7 - 818.7000000000003
# Case 8 - 730.5799999999999
# Case 9 - 802.7450000000001
# Case 10 - 874.8050000000003
# Case 11 - 739.335
# Case 12 - 809.8000000000002
# Case 13 - 803.7300000000002
# Case 14 - 804.3800000000003
# Case 15 - 812.8800000000001
# Case 16 - 740.6250000000005
# Case 17 - 747.2250000000004
# Case 18 - 784.6150000000005
# Case 19 - 741.7200000000003
# Case 20 - 841.4399999999998
# Case 21 - 831.7399999999998
# Case 22 - 797.5150000000003
# Case 23 - 858.1499999999999
# Case 24 - 786.54
# Case 25 - 772.0050000000001
# Case 26 - 715.5900000000006
# Case 27 - 821.21
# Case 28 - 789.3750000000005
# Case 29 - 709.5400000000002
# Case 30 - 663.0199999999998
# Case 31 - 799.575
# Case 32 - 812.8750000000002
# Case 33 - 852.3900000000003
# Case 34 - 816.1500000000001
# Case 35 - 709.5600000000002
# Case 36 - 825.4450000000004
# Case 37 - 800.1650000000002
# Case 38 - 809.0050000000003
# Case 39 - 804.9650000000001
# Case 40 - 776.3450000000003
# Case 41 - 724.1149999999998
# Case 42 - 783.1050000000002
# Case 43 - 675.6700000000003
# Case 44 - 734.2800000000007
# Case 45 - 709.4850000000004
# Case 46 - 812.825
# Case 47 - 907.6500000000003
# Case 48 - 854.5300000000002
# Case 49 - 810.0550000000003
# Case 50 - 776.8200000000004
# Case 51 - 797.02
# Case 52 - 885.7749999999999
# Case 53 - 784.6850000000002
# Case 54 - 840.5400000000002
# Case 55 - 728.5550000000001
# Case 56 - 750.0799999999997
# Case 57 - 745.1999999999998
# Case 58 - 876.5800000000004
# Case 59 - 763.7549999999997
# Case 60 - 820.6750000000002
# Case 61 - 825.9750000000004
# Case 62 - 763.6099999999999
# Case 63 - 819.9350000000004
# Case 64 - 703.4450000000002
# Case 65 - 753.3299999999999
# Case 66 - 800.9950000000003
# Case 67 - 814.5250000000001
# Case 68 - 850.7350000000004
# Case 69 - 761.5399999999997
# Case 70 - 766.0
# Case 71 - 720.6750000000002
# Case 72 - 801.2650000000003
# Case 73 - 845.9250000000002
# Case 74 - 834.5750000000003
# Case 75 - 819.7350000000001
# Case 76 - 902.9300000000003
# Case 77 - 757.2550000000001
# Case 78 - 739.4350000000006
# Case 79 - 734.5400000000002
# Case 80 - 837.9400000000003
# Case 81 - 773.375
# Case 82 - 800.5350000000001
# Case 83 - 747.4550000000004
# Case 84 - 765.0250000000005
# Case 85 - 825.3100000000006
# Case 86 - 804.1900000000007
# Case 87 - 856.9849999999999
# Case 88 - 721.6600000000005
# Case 89 - 768.3400000000004
# Case 90 - 788.0200000000002
# Case 91 - 799.3300000000002
# Case 92 - 851.5450000000001
# Case 93 - 915.1750000000004
# Case 94 - 769.3100000000002
# Case 95 - 770.3250000000003
# Case 96 - 829.0400000000004
# Case 97 - 744.6550000000004
# Case 98 - 700.4649999999999
# Case 99 - 818.2399999999998