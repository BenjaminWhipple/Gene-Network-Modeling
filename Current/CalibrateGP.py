"""
INPUTS:
	Dataframe for Genes of Interest: "OriginalData/Cluster_Crosswalk.csv"
   Dataframe for Gene expression: "OriginalData/GeneExpressionData_MultipleSamples.tsv"

OUTPUTS:
	GP models (pickled files): "SerializedObjects/GeneGPs/*.pkl" for general *
	Spline representation of GP mean: "SerializedObjects/MeanSplines/*.pkl" for general *
    Diagnostic Images:
        "Images/DiagnosticImages/GeneGPs/*.png" for general *
        "Images/DiagnosticImages/Splines/*.png" for general *
"""


import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import dill as pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel,RBF

# We will use N=1000 for computing the mean of the trajectory
NSAMPLES = 1000
EPS = 1e0
t = np.linspace(0,6,61)[:,np.newaxis]

# Load gene data
gene_data = pd.read_csv("OriginalData/GeneExpressionData_MultipleSamples.tsv",sep="\t")

# Load gene list
genes_list = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv")["Gene Symbol"].str.lower())

print(genes_list)

# Manage Original Data
columns = list(gene_data.columns)
MOCK_PR8_columns = [s for s in columns if "exp2_mock." in s] + [s for s in columns if "mock_PR8" in s]
RV_PR8_columns = [s for s in columns if "exp2_RV." in s] + [s for s in columns if "RV_PR8" in s]

Columns_Of_Interest = list(set(MOCK_PR8_columns+RV_PR8_columns))


all_genes = list(gene_data["GeneSymbol"])
#print(all_genes)
# Get subset of data
gene_data["GeneSymbol"]=gene_data["GeneSymbol"].str.lower()
Data_Of_Interest = gene_data[gene_data["GeneSymbol"].isin(genes_list)][Columns_Of_Interest+["GeneSymbol"]]
print(Data_Of_Interest["GeneSymbol"])

'''
For i in gene_list:
    - reshape data to have "Day" col.
    - fit GPs
    - save GPs
    - sample GPs
    - fit mean spline
    - save mean spline
    - plot data, gp, and mean spline
'''

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


for gene in genes_list:
    print(gene)
    temp=gene_data[gene_data["GeneSymbol"].isin([gene])]
    if temp.empty:
        print(f"\t No data for: {gene}")
        continue
    
    #print(temp)
    mock_temp = temp[MOCK_PR8_columns+["GeneSymbol"]]
    rv_temp = temp[RV_PR8_columns+["GeneSymbol"]]
    
    # Clean and reorganize prior to fitting GP.
    renamed_mock_temp = mock_temp.copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
    mock_temp_map = make_index_map(renamed_mock_temp.index)
    renamed_mock_temp.rename(index=mock_temp_map, inplace=True)
    
    renamed_rv_temp = rv_temp.copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
    rv_temp_map = make_index_map(renamed_rv_temp.index)
    renamed_rv_temp.rename(index=rv_temp_map, inplace=True)

    #print(renamed_mock_temp.columns)
    #print(renamed_rv_temp)    
    
    # Prepare data for fitting
    mock_t_col = renamed_mock_temp.index.to_numpy().reshape(-1,1)
    rv_t_col = renamed_rv_temp.index.to_numpy().reshape(-1,1)
    mock_obs_col = np.log10(renamed_mock_temp[f"{gene}"].to_numpy()+EPS)
    rv_obs_col = np.log10(renamed_rv_temp[f"{gene}"].to_numpy()+EPS)
    
    # Try using RBF kernel
    mock_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e1), nu=2.5) + WhiteKernel(noise_level=0.5e0, noise_level_bounds=(1e-5, 1e0))
    rv_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e1), nu=2.5) + WhiteKernel(noise_level=0.5e0, noise_level_bounds=(1e-5, 1e0))
    
    # Fit GP
    mock_this_gp = GaussianProcessRegressor(kernel=mock_kernel, n_restarts_optimizer=100)
    rv_this_gp = GaussianProcessRegressor(kernel=rv_kernel, n_restarts_optimizer=100)

    mock_this_gp.fit(mock_t_col,mock_obs_col)
    rv_this_gp.fit(rv_t_col,rv_obs_col)
    
    # Compute mean, std
    mock_mean, mock_std = mock_this_gp.predict(t,return_std=True)
    rv_mean, rv_std = rv_this_gp.predict(t,return_std=True)
    
    # Draw NSAMPLES samples
    mock_predict = mock_this_gp.sample_y(t,n_samples=NSAMPLES,random_state=42)
    rv_predict = rv_this_gp.sample_y(t,n_samples=NSAMPLES,random_state=42)
    
    # save GP predictions
    np.savetxt(f"SerializedObjects/GPSamples/MOCK_{gene}_GP_PREDICTS.csv", mock_predict, fmt='%.18e', delimiter=',', newline='\n')
    np.savetxt(f"SerializedObjects/GPSamples/RV_{gene}_GP_PREDICTS.csv", rv_predict, fmt='%.18e', delimiter=',', newline='\n')

    # Fit cubic spline to mean
    mock_mean_spline = scipy.interpolate.CubicSpline(t.flatten(),mock_mean)
    rv_mean_spline = scipy.interpolate.CubicSpline(t.flatten(),rv_mean)
    
    # save GP
    with open(f"SerializedObjects/GeneGPs/MOCK_{gene}_GP.pkl",'wb') as f:
       pickle.dump(mock_this_gp, f)

    with open(f"SerializedObjects/GeneGPs/RV_{gene}_GP.pkl",'wb') as f:
       pickle.dump(rv_this_gp, f)
       
    # save splines
    with open(f"SerializedObjects/MeanSplines/MOCK_{gene}_SPLINE.pkl",'wb') as f:
       pickle.dump(mock_mean_spline, f)

    with open(f"SerializedObjects/MeanSplines/RV_{gene}_SPLINE.pkl",'wb') as f:
       pickle.dump(rv_mean_spline, f)
    
    # Make GP plots
    fig, axs = plt.subplots(1,2,sharey=True)
    fig.suptitle(f"{gene} GP")
    axs[0].set_title(f"Mock-PR8 Group")
    axs[0].plot(t,mock_mean)
    axs[0].fill_between(t.ravel(),mock_mean-1.96*mock_std,mock_mean+1.96*mock_std,color='lightblue',alpha=0.5)
    axs[0].plot(mock_t_col,mock_obs_col,linestyle="None",marker="o")
    axs[0].set_ylabel(r"$\log_{10}$ Expression")
    axs[0].set_xlabel("DPI")
    
    axs[1].set_title(f"RV-PR8 Group")
    axs[1].plot(t,rv_mean)
    axs[1].fill_between(t.ravel(),rv_mean-1.96*rv_std,rv_mean+1.96*rv_std,color='lightblue',alpha=0.5)
    axs[1].plot(rv_t_col,rv_obs_col,linestyle="None",marker="o")
    #axs[1].ylabel(r"$\log_{10}$ Expression")
    axs[1].set_xlabel("DPI")
    plt.savefig(f"Images/DiagnosticImages/GeneGPs/{gene}_GP.png")

    # Make Spline plots
    '''
    plt.figure()
    plt.title(f"Mock-PR8 Group: {gene} Mean Spline")
    plt.plot(t,mock_mean_spline(t))
    plt.plot(mock_t_col,mock_obs_col,linestyle="None",marker="o")
    plt.ylabel(r"$\log_{10}$ Expression")
    plt.xlabel("DPI")
    plt.savefig(f"Images/DiagnosticImages/Splines/MOCK_{gene}_SPLINE.png")

    plt.figure()
    plt.title(f"RV-PR8 Group: {gene} Mean Spline")
    plt.plot(t,rv_mean_spline(t))
    plt.plot(rv_t_col,rv_obs_col,linestyle="None",marker="o")
    plt.ylabel(r"$\log_{10}$ Expression")
    plt.xlabel("DPI")
    plt.savefig(f"Images/DiagnosticImages/Splines/RV_{gene}_SPLINE.png")
    '''
