import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "./scenarios/"
ver = "v3"
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
# Case 0 - 729.0600000000004
# Case 1 - 735.8300000000004
# Case 2 - 753.605
# Case 3 - 764.2399999999998
# Case 4 - 706.335
# Case 5 - 721.0600000000004
# Case 6 - 755.6100000000006
# Case 7 - 743.6049999999998
# Case 8 - 732.5250000000003
# Case 9 - 705.5300000000002
# Case 10 - 744.3850000000002
# Case 11 - 774.6850000000009
# Case 12 - 698.1000000000001
# Case 13 - 775.9350000000002
# Case 14 - 691.1799999999998
# Case 15 - 741.6000000000001
# Case 16 - 788.3950000000007
# Case 17 - 722.81
# Case 18 - 711.0850000000005
# Case 19 - 692.8050000000005
# Case 20 - 737.6850000000002
# Case 21 - 740.3
# Case 22 - 683.7800000000002
# Case 23 - 765.5150000000003
# Case 24 - 680.72
# Case 25 - 755.7500000000002
# Case 26 - 763.9449999999997
# Case 27 - 722.5750000000005
# Case 28 - 776.0900000000006
# Case 29 - 753.8750000000005
# Case 30 - 708.6600000000001
# Case 31 - 759.9850000000004
# Case 32 - 777.7300000000005
# Case 33 - 703.8800000000001
# Case 34 - 768.3750000000007
# Case 35 - 734.7350000000004
# Case 36 - 763.1250000000002
# Case 37 - 724.2
# Case 38 - 701.1700000000003
# Case 39 - 717.6600000000005
# Case 40 - 742.1450000000002
# Case 41 - 706.8650000000002
# Case 42 - 741.8150000000005
# Case 43 - 764.0150000000006
# Case 44 - 739.8100000000006
# Case 45 - 760.8649999999998
# Case 46 - 726.5000000000002
# Case 47 - 742.1450000000002
# Case 48 - 696.3499999999999
# Case 49 - 761.9100000000001
# Case 50 - 770.3900000000003
# Case 51 - 693.5200000000007
# Case 52 - 657.165
# Case 53 - 745.6850000000004
# Case 54 - 739.3750000000005
# Case 55 - 748.1350000000002
# Case 56 - 731.6050000000002
# Case 57 - 706.8200000000004
# Case 58 - 742.2600000000002
# Case 59 - 734.415
# Case 60 - 732.8400000000001
# Case 61 - 739.7250000000006
# Case 62 - 719.9050000000002
# Case 63 - 706.1799999999998
# Case 64 - 751.7400000000007
# Case 65 - 638.1600000000001
# Case 66 - 664.5249999999999
# Case 67 - 699.3099999999997
# Case 68 - 749.6000000000004
# Case 69 - 776.2900000000004
# Case 70 - 772.5100000000007
# Case 71 - 767.5300000000002
# Case 72 - 763.0600000000004
# Case 73 - 767.0250000000005
# Case 74 - 795.6300000000006
# Case 75 - 717.6699999999996
# Case 76 - 786.4499999999998
# Case 77 - 709.1249999999998
# Case 78 - 698.74
# Case 79 - 741.7250000000006
# Case 80 - 765.3600000000001
# Case 81 - 735.3350000000003
# Case 82 - 754.49
# Case 83 - 772.2100000000003
# Case 84 - 730.8450000000007
# Case 85 - 760.9900000000002
# Case 86 - 698.1700000000005
# Case 87 - 695.99
# Case 88 - 734.3100000000004
# Case 89 - 708.3950000000002
# Case 90 - 768.8200000000006
# Case 91 - 693.8600000000008
# Case 92 - 755.8250000000007
# Case 93 - 702.9250000000002
# Case 94 - 754.8050000000003
# Case 95 - 755.8500000000004
# Case 96 - 740.6250000000005
# Case 97 - 751.0800000000004
# Case 98 - 799.2249999999999
# Case 99 - 726.655