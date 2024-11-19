import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import dill as pickle

# Load cluster crosswalk
CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
# Determine number of clusters
CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))
print(CLUSTERS)

t = np.linspace(0,6,61)[:,np.newaxis]

mock_splines = []
rv_splines = []

genes_list = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv")["Gene Symbol"].str.lower())

print(genes_list)

LAMBDA = 0.001
# Load GP Samples from Cluster Data
mock_data = []
rv_data = []

for gene in genes_list:
    with open(f"SerializedObjects/MeanSplines/MOCK_{gene}_SPLINE.pkl",'rb') as f:
       mock_spline = pickle.load(f)

    with open(f"SerializedObjects/MeanSplines/RV_{gene}_SPLINE.pkl",'rb') as f:
       rv_spline = pickle.load(f)

    mock_splines.append(mock_spline)
    rv_splines.append(rv_spline)

print("done")

#nums = [int(i) for i in CLUSTERS]

mock_preds = [mock_splines[i](t) for i in range(len(genes_list))]
mock_derivs = [mock_splines[i].derivative(1)(t) for i in range(len(genes_list))]

rv_preds = [rv_splines[i](t) for i in range(len(genes_list))]
rv_derivs = [rv_splines[i].derivative(1)(t) for i in range(len(genes_list))]

mock_preds = np.hstack(mock_preds)
mock_derivs = np.hstack(mock_derivs)

rv_preds = np.hstack(rv_preds)
rv_derivs = np.hstack(rv_derivs)

# Non sparse regression.
for i in range(len(genes_list)):
    mock_coeffs = np.linalg.pinv(mock_preds.T @ mock_preds + LAMBDA*np.eye(len(genes_list))) @ mock_preds.T @ mock_derivs[:,i]
    mock_prediction = mock_preds @ mock_coeffs
    
    rv_coeffs = np.linalg.pinv(rv_preds.T @ rv_preds + LAMBDA*np.eye(len(genes_list))) @ rv_preds.T @ rv_derivs[:,i]
    rv_prediction = rv_preds @ rv_coeffs

    plt.figure()
    plt.plot(t, mock_derivs[:,i], label="Actual")
    plt.plot(t, mock_prediction, linestyle='dashed',label="Predicted")
    plt.legend()
    plt.savefig(f"temp/mock_test_{genes_list[i]}.png")

    plt.figure()
    plt.plot(t, rv_derivs[:,i], label="Actual")
    plt.plot(t, rv_prediction, linestyle='dashed',label="Predicted")
    plt.legend()
    plt.savefig(f"temp/rv_test_{genes_list[i]}.png")
