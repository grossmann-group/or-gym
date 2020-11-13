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
    print(env.P.sum().sum())
    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))

# Cumulative Profits:
# 772.8090000000001
# 770.659
# 788.0399999999998
# 801.3960000000001
# 734.91
# 758.6170000000002
# 803.611
# 780.3389999999998
# 773.833
# 738.4200000000001
# 781.9430000000001
# 821.8660000000001
# 733.266
# 810.6200000000001
# 725.756
# 786.6999999999999
# 809.791
# 753.0849999999998
# 739.7730000000001
# 716.9159999999998
# 775.3590000000002
# 777.3160000000003
# 730.5640000000001
# 812.744
# 722.289
# 788.187
# 798.829
# 748.7810000000002
# 810.978
# 799.3689999999999
# 745.69
# 788.2110000000002
# 814.6730000000001
# 746.0129999999999
# 805.2479999999999
# 774.2210000000001
# 802.3559999999999
# 767.704
# 742.1680000000001
# 753.1870000000001
# 778.784
# 743.916
# 780.9660000000002
# 806.8139999999999
# 785.485
# 797.791
# 763.4740000000003
# 778.962
# 734.0190000000001
# 792.6450000000001
# 791.869
# 730.803
# 696.956
# 782.573
# 776.7180000000001
# 785.0770000000002
# 766.641
# 745.0999999999999
# 777.451
# 776.028
# 769.9190000000001
# 777.3910000000002
# 759.0620000000004
# 743.3629999999999
# 788.1900000000002
# 701.8499999999999
# 731.4950000000001
# 791.174
# 808.4960000000002
# 809.6529999999999
# 804.4989999999999
# 810.6389999999999
# 799.5029999999999
# 802.6550000000001
# 754.893
# 810.518
# 749.2270000000001
# 744.079
# 778.8590000000002
# 788.424
# 764.1940000000002
# 795.684
# 815.5590000000002
# 753.5290000000001
# 793.602
# 726.745
# 732.7460000000002
# 771.5680000000002
# 750.866
# 803.359
# 730.7660000000001
# 793.081
# 738.793
# 789.7270000000001
# 793.2430000000002
# 771.215
# 790.137
# 827.1250000000001
# 761.4630000000002