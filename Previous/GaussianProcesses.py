"""
Outline
- Load files
- Prototype for loop on one dataset

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# Load data, rename index column to "Day"

IFN_ALPHA_MOCK = pd.read_csv("ProcessedData/IFN-Alpha_Mock-PR8").rename(columns={"Unnamed: 0": "Day"})
IFN_ALPHA_RV = pd.read_csv("ProcessedData/IFN-Alpha_RV-PR8").rename(columns={"Unnamed: 0": "Day"})
IFN_GAMMA_MOCK = pd.read_csv("ProcessedData/IFN-Gamma_Mock-PR8").rename(columns={"Unnamed: 0": "Day"})
IFN_GAMMA_RV = pd.read_csv("ProcessedData/IFN-Gamma_RV-PR8").rename(columns={"Unnamed: 0": "Day"})


DATA = [IFN_ALPHA_MOCK,IFN_ALPHA_RV,IFN_GAMMA_MOCK,IFN_GAMMA_RV]
NAMES = ["IFN_ALPHA_MOCK","IFN_ALPHA_RV","IFN_GAMMA_MOCK","IFN_GAMMA_RV"]

SAMPLES = 1000
SAMPLED_TIMES = np.linspace(0,6,301)[:,np.newaxis]

np.savetxt("SurrogateSamples/SAMPLE_TIMES.txt",np.linspace(0,6,301),delimiter=",")

kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e4), nu=1.5) + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-5, 1e4))

start = time.time()

# Get all columns not "Day"
for k in range(len(DATA)):
    preds = []
    models = []
    cols = [i for i in DATA[k] if i != "Day"]
    #cols = cols[0:10]
    i = 0
    for col in cols:
        i += 1
        print(f"{i}/{len(cols)}")
        print(col)
        t_col = DATA[k]["Day"].to_numpy()
        obs_col = np.clip(DATA[k][col].to_numpy(),a_min=1e-8,a_max=None)
        this_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
        this_gp.fit(t_col.reshape(-1,1),np.log10(obs_col))
        models.append(this_gp)
    
    for i in range(len(cols)):
        sample = models[i].sample_y(SAMPLED_TIMES, SAMPLES,0)
        clipped_sample = np.clip(sample,a_min=1e-8,a_max=None)
        max_vals = np.max(clipped_sample,axis=0)
        preds.append(clipped_sample/max_vals)
    
    aggregates = np.zeros(preds[0].shape)
    for i in range(SAMPLES):
        temp = np.zeros((len(SAMPLED_TIMES), len(cols)))
        for j in range(len(preds)):
            temp[:,j]=np.log(preds[j][:,i])
        
        # We aggregate by taking the geometric mean across the different genes
        this_agg = np.exp(np.mean(temp,axis=1))
        this_max = np.max(this_agg,axis=0)
        
        aggregates[:,i]=this_agg/this_max
    
    np.savetxt(f"SurrogateSamples/{NAMES[k]}.txt",aggregates,delimiter=",")
    
    plt.figure(figsize=(4.5,3.5))
    #plt.title(f"{NAMES[k]}")
    plt.plot(SAMPLED_TIMES, np.clip(aggregates,a_min=0.5,a_max=None), color="cornflowerblue",alpha=0.01)
    plt.plot(SAMPLED_TIMES, np.mean(aggregates,axis=1),color="black")
    plt.ylabel("Aggregated Percent Expression")
    plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    plt.xlabel("Days")
    plt.savefig(f"Plots/{NAMES[k]}.png",dpi=300)

end = time.time()
print(f"Time taken: {end-start}")