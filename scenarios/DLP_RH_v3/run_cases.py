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
    print(env.P.sum().sum())
    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

# Cumulative Profits:
# 705.3950000000002
# 705.8300000000005
# 723.4700000000003
# 734.1200000000003
# 676.3550000000004
# 690.8750000000002
# 736.0500000000002
# 713.46
# 710.9850000000004
# 681.8
# 714.2600000000002
# 759.6450000000001
# 667.9400000000005
# 745.845
# 660.8950000000004
# 724.135
# 758.3100000000002
# 692.62
# 681.05
# 662.735
# 707.5200000000002
# 710.1300000000002
# 662.0449999999998
# 756.6500000000002
# 650.4250000000002
# 725.7600000000002
# 733.9550000000004
# 692.5450000000001
# 746.0100000000001
# 736.6750000000002
# 678.495
# 729.8550000000004
# 754.0349999999999
# 677.94
# 738.29
# 704.6650000000003
# 735.19
# 700.6299999999999
# 675.1500000000001
# 687.5000000000002
# 716.1950000000004
# 676.6750000000003
# 713.8350000000003
# 746.7050000000004
# 718.2450000000003
# 730.77
# 696.3750000000003
# 712.0150000000001
# 666.1300000000001
# 731.8400000000001
# 740.325
# 663.3400000000006
# 633.335
# 715.5399999999997
# 709.3300000000002
# 718.2150000000004
# 701.4900000000005
# 676.6200000000002
# 712.2500000000002
# 708.565
# 702.585
# 709.565
# 689.5300000000002
# 676.0200000000002
# 721.5
# 634.3449999999999
# 669.1150000000001
# 725.9100000000001
# 746.2550000000002
# 742.4500000000005
# 737.4700000000001
# 743.5250000000002
# 736.9050000000002
# 765.4900000000002
# 687.5250000000001
# 756.4500000000003
# 680.9750000000003
# 676.99
# 711.6250000000001
# 735.1400000000003
# 705.155
# 732.96
# 748.4350000000002
# 700.7250000000001
# 730.9050000000004
# 668.0850000000002
# 665.6200000000001
# 704.22
# 682.4449999999999
# 738.7199999999998
# 663.5050000000003
# 725.7350000000002
# 672.8050000000003
# 724.8150000000003
# 725.7800000000002
# 710.4850000000002
# 723.11
# 769.2450000000001
# 696.4250000000002