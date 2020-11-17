import or_gym
import pickle
from pyomo.opt import SolverFactory
from or_gym.algos.supply_network.math_prog import *
from or_gym.algos.supply_network.stoch_prog import *

#imputs
path = "../../scenarios/"
ver = "v2"
model = "DLP"
mode = "RH"
window = 10
solver = "gurobi"

#create file names
filename = path+"in"+ver+".pkl"
D = pickle.load(open(filename,'rb'))
filesave = path+model+"_"+mode+"_"+ver+"/"

#solve shrinking horizon model
#solve shrinking horizon model
total_profit_record = []
non_optimal_record = []
cpu_time_record = []
wall_time_record = []
import time

for i in range(100):
    start_time = time.time()
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
        if str(res['Solver'][0]['Termination condition']) != 'optimal':
            print("ERROR: NOT OPTIMAL")
            break
        #extract reorder quantities
        Ropt=m.R.get_values()
        #pass action for the current timestep only (i.e. t=0)
        action={e[1:]:Ropt[e] for e in Ropt.keys() if e[0]==0}
        outcome=env.step(action)
        cpu_time_record.append(res['Solver'][0]['Time'])
    #print profit for the total simulation run
    print(env.P.sum().sum())
    total_profit_record.append(env.P.sum().sum())
    end_time = time.time()
    wall_time_record.append(end_time-start_time)

    #pickle environment
    pickle.dump(env, open(filesave+"run"+str(i)+".pkl","wb"))
    # break 
    # if i >=5:
    #     break 
print(filesave)

cpu_time_record_iter = []
for i in range(100):
    cpu_time_record_iter.append(0)
    for j in range(10):
        cpu_time_record_iter[i] += cpu_time_record[i*30+j]

# Cumulative Profits:
# 684.285
# 839.3150000000003
# 724.3500000000001
# 810.2450000000001
# 840.6000000000003
# 741.8250000000002
# 829.315
# 788.65
# 700.3450000000003
# 772.5800000000002
# 844.6800000000001
# 711.2800000000002
# 779.7050000000002
# 773.5500000000002
# 785.0800000000002
# 783.1550000000001
# 731.6
# 717.0750000000003
# 754.4350000000003
# 711.4650000000001
# 811.3050000000001
# 801.5800000000002
# 767.6400000000001
# 827.8800000000001
# 756.4550000000003
# 741.7500000000005
# 685.3700000000002
# 791.0750000000002
# 761.325
# 679.3200000000002
# 649.7600000000001
# 769.4300000000002
# 785.0450000000002
# 822.4400000000002
# 785.8650000000002
# 679.2850000000003
# 795.4150000000002
# 770.2650000000001
# 778.7800000000004
# 774.8150000000002
# 752.3300000000002
# 693.9250000000001
# 752.9550000000004
# 645.4750000000003
# 704.1000000000003
# 679.2650000000001
# 789.1600000000001
# 877.6050000000004
# 824.37
# 779.9450000000004
# 768.3100000000002
# 766.9200000000001
# 855.6350000000001
# 754.5250000000002
# 810.435
# 698.325
# 719.8850000000004
# 715.0450000000004
# 846.3500000000004
# 733.595
# 790.6100000000001
# 795.9250000000002
# 733.3900000000003
# 793.9750000000003
# 673.3499999999998
# 723.0600000000003
# 771.425
# 788.6150000000001
# 824.8300000000003
# 739.72
# 740.1399999999999
# 690.5800000000004
# 771.1700000000002
# 822.2650000000001
# 804.4500000000002
# 789.6100000000004
# 872.9400000000003
# 729.2100000000002
# 711.3900000000001
# 706.47
# 808.085
# 743.25
# 770.4100000000001
# 717.3249999999999
# 734.8750000000003
# 805.7950000000001
# 774.0550000000001
# 826.7750000000001
# 695.6800000000003
# 738.1650000000001
# 757.7800000000003
# 769.2750000000001
# 821.4750000000003
# 885.1650000000003
# 739.2550000000003
# 742.2650000000001
# 798.8550000000001
# 714.5100000000002
# 670.1650000000002
# 796.5550000000003