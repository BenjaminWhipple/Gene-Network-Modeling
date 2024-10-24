"""
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

### Prepare data

gene_data = pd.read_csv("OriginalData/OldGeneExpressionData_MultipleSamples.tsv",sep="\t")

GenesOfInterest = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv",sep=",")["Gene Symbol"])

print(GenesOfInterest)

columns = list(gene_data.columns)


MOCK_PR8_columns = ["GeneSymbol"] + [s for s in columns if "mock." in s] + [s for s in columns if "mock_PR8" in s]
RV_PR8_columns = ["GeneSymbol"] + [s for s in columns if "RV." in s] + [s for s in columns if "RV_PR8" in s]

Gene_Data = gene_data[gene_data["GeneSymbol"].isin(GenesOfInterest)]

MOCK_PR8 = Gene_Data[MOCK_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
RV_PR8 = Gene_Data[RV_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()

# This function is highly specific to our data.
def make_index_map(a_list):
    a_map = {}
    for i in a_list:
        if ".0" in i:
            a_map[i]=0
        elif ".2" in i:
            a_map[i]=2
        elif ".4" in i:
            a_map[i]=4
        elif ".6" in i:
            a_map[i]=6
        else:
            pass
    
    return a_map

MOCK_PR8_map = make_index_map(MOCK_PR8.index)
MOCK_PR8.rename(index=MOCK_PR8_map, inplace=True)

RV_PR8_map = make_index_map(RV_PR8.index)
RV_PR8.rename(index=RV_PR8_map, inplace=True)

MOCK_PR8.to_csv("ProcessedData/Mock-PR8")
RV_PR8.to_csv("ProcessedData/RV-PR8")


### Plot Gaussian Processes individually.

# Load data, rename index column to "Day"

DATA = pd.read_csv("ProcessedData/Mock-PR8").rename(columns={"Unnamed: 0": "Day"})

SAMPLES = 1000
SAMPLED_TIMES = np.linspace(0,6,301)[:,np.newaxis]

np.savetxt("SurrogateSamples/SAMPLE_TIMES.txt",np.linspace(0,6,301),delimiter=",")

kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e4), nu=1.5) + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-5, 1e4))

start = time.time()

# Get all columns not "Day"
preds = []
models = []
cols = [i for i in DATA if i != "Day"]
#cols = cols[0:10]
i = 0

#NUM=5

#np.savetxt(f"column_names.txt",np.array(cols))

print(cols)

GROUP = "Mock-PR8"
for col in cols:
    i += 1
    print(f"{i}/{len(cols)}")
    print(col)
    t_col = DATA["Day"].to_numpy()
    obs_col = np.clip(DATA[col].to_numpy(),a_min=1e-8,a_max=None)
    this_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    this_gp.fit(t_col.reshape(-1,1),np.log10(obs_col))
    models.append(this_gp)

for i in range(len(cols)):
    sample = models[i].sample_y(SAMPLED_TIMES, SAMPLES,0)
    clipped_sample = np.clip(sample,a_min=1e-8,a_max=None)
    max_vals = np.max(clipped_sample,axis=0)
    preds.append(clipped_sample/max_vals)
    
    #print(np.array(preds))
    #print(np.array(preds).shape)
    
    # We also want to save predictions to numpy.
    np.savetxt(f"SurrogateSamples/{GROUP}_{cols[i]}.txt",np.array(clipped_sample),delimiter=",")
    """
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data: {GROUP}")
    #plt.yscale("log")
    plt.plot(DATA["Day"],np.log10(DATA[cols[i]]),"o",color="tomato")
    plt.xlabel("Days")
    plt.ylabel("Expression (log10 read counts)")
    plt.savefig(f"Images/ComponentPlots/Mock-PR8_{cols[i]}_data.png")
    
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data with gp: {GROUP}")
    plt.plot(SAMPLED_TIMES,sample,alpha=0.01,color="cornflowerblue")
    plt.plot(SAMPLED_TIMES, np.mean(sample,axis=1),color="black")
    plt.plot(DATA["Day"],np.log10(DATA[cols[i]]),"o",color="tomato")
    plt.xlabel("Days")
    plt.ylabel("Expression (log10 read counts)")
    plt.savefig(f"Images/ComponentPlots/Mock-PR8_{cols[i]}_data_with_gp.png")
    
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data with gp normalized: {GROUP}")
    plt.plot(SAMPLED_TIMES,clipped_sample/max_vals,alpha=0.01,color="cornflowerblue")
    plt.plot(SAMPLED_TIMES, np.mean(clipped_sample/max_vals,axis=1),color="black")
    plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    plt.xlabel("Days")
    plt.ylabel("Percent Expression")
    plt.savefig(f"Images/ComponentPlots/Mock-PR8_{cols[i]}_data_with_gp_normalized.png",dpi=300)
    """
# NOW repeat for RV-PR8
DATA = pd.read_csv("ProcessedData/RV-PR8").rename(columns={"Unnamed: 0": "Day"})

SAMPLES = 1000
SAMPLED_TIMES = np.linspace(0,6,301)[:,np.newaxis]

np.savetxt("SurrogateSamples/SAMPLE_TIMES.txt",np.linspace(0,6,301),delimiter=",")

kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e4), nu=1.5) + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-5, 1e4))

start = time.time()

# Get all columns not "Day"
preds = []
models = []
cols = [i for i in DATA if i != "Day"]
#cols = cols[0:10]
i = 0

#NUM=5

GROUP = "RV-PR8"

for col in cols:
    i += 1
    print(f"{i}/{len(cols)}")
    print(col)
    t_col = DATA["Day"].to_numpy()
    obs_col = np.clip(DATA[col].to_numpy(),a_min=1e-8,a_max=None)
    this_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    this_gp.fit(t_col.reshape(-1,1),np.log10(obs_col))
    models.append(this_gp)

for i in range(len(cols)):
    sample = models[i].sample_y(SAMPLED_TIMES, SAMPLES,0)
    clipped_sample = np.clip(sample,a_min=1e-8,a_max=None)
    max_vals = np.max(clipped_sample,axis=0)
    preds.append(clipped_sample/max_vals)
    
    # We also want to save predictions to numpy.
    np.savetxt(f"SurrogateSamples/{GROUP}_{cols[i]}.txt",np.array(clipped_sample))
    """
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data: {GROUP}")
    #plt.yscale("log")
    plt.plot(DATA["Day"],np.log10(DATA[cols[i]]),"o",color="tomato")
    plt.xlabel("Days")
    plt.ylabel("Expression (log10 read counts)")
    plt.savefig(f"Images/ComponentPlots/RV-PR8_{cols[i]}_data.png")
    
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data with gp: {GROUP}")
    plt.plot(SAMPLED_TIMES,sample,alpha=0.01,color="cornflowerblue")
    plt.plot(SAMPLED_TIMES, np.mean(sample,axis=1),color="black")
    plt.plot(DATA["Day"],np.log10(DATA[cols[i]]),"o",color="tomato")
    plt.xlabel("Days")
    plt.ylabel("Expression (log10 read counts)")
    plt.savefig(f"Images/ComponentPlots/RV-PR8_{cols[i]}_data_with_gp.png")
    
    plt.figure(figsize=(4.5,3.5))
    plt.title(f"{cols[i]} data with gp normalized: {GROUP}")
    plt.plot(SAMPLED_TIMES,clipped_sample/max_vals,alpha=0.01,color="cornflowerblue")
    plt.plot(SAMPLED_TIMES, np.mean(clipped_sample/max_vals,axis=1),color="black")
    plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    plt.xlabel("Days")
    plt.ylabel("Percent Expression")
    plt.savefig(f"Images/ComponentPlots/RV-PR8_{cols[i]}_data_with_gp_normalized.png",dpi=300)
    """
