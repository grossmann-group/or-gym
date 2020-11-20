import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import pickle

path = "c:\\Users\\HD\\Documents\\Development\\or-gym\\scenarios\\"
# path = "./scenarios/"
ver = "v3"
model = "MSSP_SH"
pickle_path = path+model+"_"+ver+"/run"

def on_hand_inventory():
    inv = {j+1: pd.DataFrame(np.zeros((31,100))) for j in range(6)}
    for i in range(100):
        env = pickle.load(open(pickle_path+str(i)+".pkl", "rb"))
        for j in range(6):
            inv[j+1][i] = env.X[j+1]
    
    inv_mean = {j+1: inv[j+1].mean(axis=1) for j in range(6)}

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)
    axs = axs.ravel()
    for j in range(6):
        axs[j].plot(inv_mean[j+1])
        axs[j].set_title("Node = "+str(j+1))

# tot_inv = {j+1: pd.DataFrame(np.zeros((31,100))) for j in range(6)}
# for i in range(100):
#     env = pickle.load(open(pickle_path+str(i)+".pkl", "rb"))
#     for j in range(6):
#         tot_inv[j+1][i] = env.X[j+1] #+ env.Y[[(k,j+1) for k in env.graph.predecessors(j+1)]].sum(axis=1)

# tot_inv_mean = {j+1: tot_inv[j+1].mean(axis=1) for j in range(6)}

# fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)
# axs = axs.ravel()
# for j in range(6):
#     axs[j].plot(tot_inv_mean[j+1])
#     axs[j].set_title("Node = "+str(j+1))

tot_inv = {j+1: pd.DataFrame(np.zeros((31,100))) for j in range(6)}
lost_sales = np.zeros((30,100))
for i in range(100):
    env = pickle.load(open(pickle_path+str(i)+".pkl", "rb"))
    lost_sales[:,i] = 2.100*env.U[(1,0)]
    for j in range(6):
        tot_inv[j+1][i] = env.graph.nodes[j+1]['h']*env.X[j+1] + np.sum([env.graph.edges[k,j+1]['g']*env.Y[(k,j+1)] for k in env.graph.predecessors(j+1)], axis=0)

# tot_inv_mean = {j+1: tot_inv[j+1].mean().mean() for j in range(6)}

inv_costs = sum(tot_inv[j+1] for j in range(6))
inv_costs_mean = inv_costs.mean(axis=1).sum()

lost_costs_mean = np.mean(np.sum(lost_sales, axis=0))
print(inv_costs_mean)
print(lost_costs_mean)

# fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)
# axs = axs.ravel()
# for j in range(6):
#     axs[j].plot(tot_inv_mean[j+1])
#     axs[j].set_title("Node = "+str(j+1))
