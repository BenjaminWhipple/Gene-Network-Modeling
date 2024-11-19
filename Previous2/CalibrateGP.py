"""
INPUTS:
	Dataframe for cluster crosswalk: "ProcessedData/Cluster_Crosswalk.csv"
	Dataframes for each cluster between RV-PR8 and MOCK-PR8 groups: "ProcessedData/ClusterMeasurements/*.csv" for general *.

OUTPUTS:
	GP models (pickled files): "SerializedObjects/ClusterGPs/*.pkl" for general *
	GP predictions: "SerializedObjects/ClusterGPPredicts/*.csv" for general *
	Spline representation of GP mean: "SerializedObjects/MeanSplines/*.pkl" for general *
    Diagnostic Images:
        "Images/DiagnosticImages/GPClusters/*.png" for general *
        "Images/DiagnosticImages/Splines/*.png" for general *
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import dill as pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# We will use N=1000 for computing the mean of the trajectory
NSAMPLES = 1000
EPS = 1e0
t = np.linspace(0,6,61)[:,np.newaxis]

# Load cluster crosswalk
CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
# Determine number of clusters
CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))
print(CLUSTERS)

for i in CLUSTERS:
    print(i)
    mock_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e4), nu=2.5) + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-5, 1e4))
    rv_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e4), nu=2.5) + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-5, 1e4))
    
    # Read in cluster data
    cluster_i_mock = pd.read_csv(f"ProcessedData/ClusterMeasurements/MOCK_PR8_Cluster_{int(i)}_Measurements.csv")[["Day",f"Cluster_{int(i)}_Expression"]]
    cluster_i_rv = pd.read_csv(f"ProcessedData/ClusterMeasurements/RV_PR8_Cluster_{int(i)}_Measurements.csv")[["Day",f"Cluster_{int(i)}_Expression"]]
    
    # Prepare data for fitting
    mock_t_col = cluster_i_mock["Day"].to_numpy().reshape(-1,1)
    rv_t_col = cluster_i_rv["Day"].to_numpy().reshape(-1,1)

    mock_obs_col = np.log10(cluster_i_mock[f"Cluster_{int(i)}_Expression"].to_numpy()+EPS)
    rv_obs_col = np.log10(cluster_i_rv[f"Cluster_{int(i)}_Expression"].to_numpy()+EPS)

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

    # Fit cubic spline to mean
    mock_mean_spline = scipy.interpolate.CubicSpline(t.flatten(),mock_mean)
    rv_mean_spline = scipy.interpolate.CubicSpline(t.flatten(),rv_mean)

    # save GP
    with open(f"SerializedObjects/ClusterGPs/MOCK_PR8_CLUSTER_{i}_GP.pkl",'wb') as f:
       pickle.dump(mock_this_gp, f)

    with open(f"SerializedObjects/ClusterGPs/RV_PR8_CLUSTER_{i}_GP.pkl",'wb') as f:
       pickle.dump(rv_this_gp, f)

    # save GP predictions
    np.savetxt(f"SerializedObjects/ClusterGPPredicts/MOCK_PR8_CLUSTER_{i}_GP_PREDICTS.csv", mock_predict, fmt='%.18e', delimiter=',', newline='\n')
    np.savetxt(f"SerializedObjects/ClusterGPPredicts/RV_PR8_CLUSTER_{i}_GP_PREDICTS.csv", rv_predict, fmt='%.18e', delimiter=',', newline='\n')

    # save splines
    with open(f"SerializedObjects/MeanSplines/MOCK_PR8_CLUSTER_{i}_SPLINE.pkl",'wb') as f:
       pickle.dump(mock_mean_spline, f)

    with open(f"SerializedObjects/MeanSplines/RV_PR8_CLUSTER_{i}_SPLINE.pkl",'wb') as f:
       pickle.dump(rv_mean_spline, f)

    # Make GP plots
    plt.figure()
    plt.plot(t,mock_mean)
    plt.fill_between(t.ravel(),mock_mean-1.96*mock_std,mock_mean+1.96*mock_std,color='lightblue',alpha=0.5)
    plt.plot(mock_t_col,mock_obs_col,linestyle="None",marker="o")
    plt.plot()
    #plt.yscale("log")
    plt.savefig(f"Images/DiagnosticImages/GPClusters/MOCK_PR8_CLUSTER_{i}_GP.png")

    plt.figure()
    plt.plot(t,rv_mean)
    plt.fill_between(t.ravel(),rv_mean-1.96*rv_std,rv_mean+1.96*rv_std,color='lightblue',alpha=0.5)
    plt.plot(rv_t_col,rv_obs_col,linestyle="None",marker="o")
    plt.plot()
    #plt.yscale("log")
    plt.savefig(f"Images/DiagnosticImages/GPClusters/RV_PR8_CLUSTER_{i}_GP.png")

    # Make Spline plots
    plt.figure()
    plt.plot(t,mock_mean_spline(t))
    plt.plot(mock_t_col,mock_obs_col,linestyle="None",marker="o")
    plt.plot()
    plt.savefig(f"Images/DiagnosticImages/Splines/MOCK_PR8_CLUSTER_{i}_SPLINE.png")

    plt.figure()
    plt.plot(t,rv_mean_spline(t))
    plt.plot(rv_t_col,rv_obs_col,linestyle="None",marker="o")
    plt.plot()
    plt.savefig(f"Images/DiagnosticImages/Splines/RV_PR8_CLUSTER_{i}_SPLINE.png")