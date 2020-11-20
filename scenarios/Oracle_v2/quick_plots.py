import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import pickle

path = "c:\\Users\\HD\\Documents\\Development\\or-gym\\scenarios\\"
# path = "./scenarios/"
ver = "v2"
model = "Oracle"
pickle_path = path+model+"_"+ver+"/run"

tot_inv = {j+1: pd.DataFrame(np.zeros((31,100))) for j in range(6)}
for i in range(100):
    env = pickle.load(open(pickle_path+str(i)+".pkl", "rb"))
    for j in range(6):
        tot_inv[j+1][i] = env.X[j+1] #+ env.Y[[(k,j+1) for k in env.graph.predecessors(j+1)]].sum(axis=1)

tot_inv_mean = {j+1: tot_inv[j+1].mean(axis=1) for j in range(6)}

fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)
axs = axs.ravel()
for j in range(6):
    axs[j].plot(tot_inv_mean[j+1])
    axs[j].set_title("Node = "+str(j+1))