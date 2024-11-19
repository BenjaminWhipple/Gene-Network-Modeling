"""
ABOUT:
In this file we try to express the derivative of a spline with respect to the levels of different the splines. 
That is, we try to model the time evolution for the system as a linear model.

INPUTS:
    

OUTPUTS:


NOTES:
This file violates DRY fairly heavily. 
Refactoring would convert the main for loop into a function. 
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import dill as pickle


# Define globals
t = np.linspace(0,6,61)[:,np.newaxis]

# Load Cluster Crosswalk
CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))

# Load GP Samples from Cluster Data
mock_data = []
rv_data = []

for i in CLUSTERS:
    mock_data_i = np.loadtxt(f"SerializedObjects/ClusterGPPredicts/MOCK_PR8_CLUSTER_{i}_GP_PREDICTS.csv",delimiter=",")
    rv_data_i = np.loadtxt(f"SerializedObjects/ClusterGPPredicts/RV_PR8_CLUSTER_{i}_GP_PREDICTS.csv",delimiter=",")

    mock_data.append(mock_data_i)
    rv_data.append(rv_data_i)


# Conduct estimation procedure
NSAMPLES = mock_data[0].shape[1]
#NSAMPLES = 3

mock_coefficients = []
mock_predictions = []
mock_solution_predictions = []

rv_coefficients = []
rv_predictions = []
rv_solution_predictions = []

for i in range(NSAMPLES):
    print(i)
    mock_splines = []
    rv_splines = []

    # Compute splines. 
    for j in CLUSTERS:
        # Select data
        mock_data_j = mock_data[j][:,i]
        rv_data_j = rv_data[j][:,i]

        # Spline interpolation
        mock_spline_j = scipy.interpolate.CubicSpline(t.flatten(),mock_data_j)
        rv_spline_j = scipy.interpolate.CubicSpline(t.flatten(),rv_data_j)

        mock_splines.append(mock_spline_j)
        rv_splines.append(rv_spline_j)

    # Aggregate observations and derivatives
    mock_preds = [mock_splines[k](t) for k in CLUSTERS]
    mock_derivs = [mock_splines[k].derivative(1)(t) for k in CLUSTERS]
    mock_preds = np.hstack(mock_preds)
    mock_derivs = np.hstack(mock_derivs)

    rv_preds = [rv_splines[k](t) for k in CLUSTERS]
    rv_derivs = [rv_splines[k].derivative(1)(t) for k in CLUSTERS]
    rv_preds = np.hstack(rv_preds)
    rv_derivs = np.hstack(rv_derivs)

    # Compute coefficients and predictions
    this_mock_coeffs = []
    this_mock_pred = []
    this_mock_soln = []

    this_rv_coeffs = []
    this_rv_pred = []
    this_rv_soln = []

    for j in CLUSTERS:
        mock_coeffs_j = np.linalg.pinv(mock_preds.T @ mock_preds) @ mock_preds.T @ mock_derivs[:,j]
        mock_prediction_j = mock_preds @ mock_coeffs_j
        mock_soln_j = np.cumsum(mock_prediction_j*(t[1]-t[0]))+mock_preds[0,j]

        rv_coeffs_j = np.linalg.pinv(rv_preds.T @ rv_preds) @ rv_preds.T @ rv_derivs[:,j]
        rv_prediction_j = rv_preds @ rv_coeffs_j
        rv_soln_j = np.cumsum(rv_prediction_j*(t[1]-t[0]))+rv_preds[0,j]

        this_mock_coeffs.append(mock_coeffs_j)
        this_mock_pred.append(mock_prediction_j)
        this_mock_soln.append(mock_soln_j)

        this_rv_coeffs.append(rv_coeffs_j)
        this_rv_pred.append(rv_prediction_j)
        this_rv_soln.append(mock_soln_j)

    this_mock_coeffs = np.vstack(this_mock_coeffs)
    this_mock_pred = np.vstack(this_mock_pred)
    this_mock_soln = np.vstack(this_mock_soln)
    
    this_rv_coeffs = np.vstack(this_rv_coeffs)
    this_rv_pred = np.vstack(this_rv_pred)
    this_rv_soln = np.vstack(this_rv_soln)    

    mock_coefficients.append(this_mock_coeffs)
    mock_predictions.append(this_mock_pred)
    mock_solution_predictions.append(this_mock_soln)
    
    rv_coefficients.append(this_rv_coeffs)
    rv_predictions.append(this_rv_pred)
    rv_solution_predictions.append(this_rv_soln)

print(mock_coefficients[0])
print(rv_coefficients[0])

# Write results to files
with open("SerializedObjects/EstimationResults/MOCK_PR8_PREDICTIONS.pkl",'wb') as f:
    pickle.dump(mock_predictions, f)
with open("SerializedObjects/EstimationResults/MOCK_PR8_SOLUTIONS.pkl",'wb') as f:
    pickle.dump(mock_solution_predictions, f)
with open("SerializedObjects/EstimationResults/MOCK_PR8_COEFFICIENTS.pkl",'wb') as f:
    pickle.dump(mock_coefficients, f)

with open("SerializedObjects/EstimationResults/RV_PR8_PREDICTIONS.pkl",'wb') as f:
    pickle.dump(rv_predictions, f)
with open("SerializedObjects/EstimationResults/RV_PR8_SOLUTIONS.pkl",'wb') as f:
    pickle.dump(rv_solution_predictions, f)
with open("SerializedObjects/EstimationResults/RV_PR8_COEFFICIENTS.pkl",'wb') as f:
    pickle.dump(rv_coefficients, f)