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
    print(env.P.sum().sum())
    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

# Cumulative Profits:
# 752.555
# 887.8650000000001
# 789.418
# 845.5489999999999
# 844.313
# 809.971
# 843.0830000000001
# 826.7320000000001
# 767.5959999999999
# 827.916
# 846.7880000000001
# 779.0380000000001
# 838.388
# 833.1
# 847.7810000000002
# 770.9240000000001
# 800.12
# 784.0580000000001
# 812.4990000000001
# 779.1730000000001
# 870.964
# 858.302
# 830.702
# 855.6350000000001
# 823.995
# 808.15
# 753.112
# 845.302
# 817.714
# 746.8870000000002
# 718.1100000000001
# 837.1070000000001
# 825.6320000000001
# 774.8730000000003
# 836.6470000000002
# 746.8880000000001
# 826.08
# 822.2290000000002
# 828.0169999999999
# 829.275
# 820.231
# 761.5369999999999
# 805.5970000000001
# 713.187
# 771.584
# 744.9060000000001
# 847.475
# 827.5100000000001
# 855.0490000000001
# 832.3530000000001
# 817.863
# 833.805
# 896.5269999999999
# 818.4690000000002
# 823.2679999999999
# 766.3830000000002
# 787.034
# 778.447
# 830.336
# 788.185
# 842.8650000000001
# 848.043
# 800.6710000000002
# 825.323
# 735.1530000000002
# 797.807
# 838.4430000000002
# 865.2480000000002
# 800.7610000000001
# 802.7620000000001
# 733.8370000000002
# 812.854
# 845.0850000000002
# 823.8520000000001
# 844.1820000000001
# 851.8770000000001
# 796.27
# 779.442
# 774.164
# 791.1000000000001
# 810.332
# 820.984
# 785.0839999999998
# 801.6790000000001
# 859.3420000000001
# 831.2300000000001
# 854.3530000000002
# 762.618
# 804.5680000000001
# 816.7049999999999
# 818.905
# 835.998
# 839.667
# 783.6100000000001
# 808.859
# 852.0870000000001
# 782.0840000000002
# 738.0520000000001
# 856.9970000000001