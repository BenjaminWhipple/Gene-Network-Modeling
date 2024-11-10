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

for i in CLUSTERS:
    print(i)
    with open(f"SerializedObjects/MeanSplines/MOCK_PR8_CLUSTER_{i}_SPLINE.pkl",'rb') as f:
       mock_spline = pickle.load(f)

    with open(f"SerializedObjects/MeanSplines/RV_PR8_CLUSTER_{i}_SPLINE.pkl",'rb') as f:
       rv_spline = pickle.load(f)

    mock_splines.append(mock_spline)
    rv_splines.append(rv_spline)

print("done")

nums = [int(i) for i in CLUSTERS]

mock_preds = [mock_splines[i](t) for i in CLUSTERS]
mock_derivs = [mock_splines[i].derivative(1)(t) for i in CLUSTERS]

rv_preds = [rv_splines[i](t) for i in CLUSTERS]
rv_derivs = [rv_splines[i].derivative(1)(t) for i in CLUSTERS]

mock_preds = np.hstack(mock_preds)
mock_derivs = np.hstack(mock_derivs)

rv_preds = np.hstack(rv_preds)
rv_derivs = np.hstack(rv_derivs)

# Non sparse regression.
for i in CLUSTERS:
    mock_coeffs = np.linalg.pinv(mock_preds.T @ mock_preds) @ mock_preds.T @ mock_derivs[:,i]
    mock_prediction = mock_preds @ mock_coeffs
    
    rv_coeffs = np.linalg.pinv(rv_preds.T @ rv_preds) @ rv_preds.T @ rv_derivs[:,i]
    rv_prediction = rv_preds @ rv_coeffs

    plt.figure()
    plt.plot(t, mock_prediction, label="Predicted")
    plt.plot(t, mock_derivs[:,i], label="Actual")
    plt.legend()
    plt.savefig(f"temp/mock_test_{i}.png")

    plt.figure()
    plt.plot(t, rv_prediction, label="Predicted")
    plt.plot(t, rv_derivs[:,i], label="Actual")
    plt.legend()
    plt.savefig(f"temp/rv_test_{i}.png")
