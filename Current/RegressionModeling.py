
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import dill as pickle


# Define globals
t = np.linspace(0,6,61)[:,np.newaxis]

# Load Cluster Crosswalk
#CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
#CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))

# Load Genes
genes_list = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv")["Gene Symbol"].str.lower())

print(genes_list)

LAMBDA = 0.001
# Load GP Samples from Cluster Data
mock_data = []
rv_data = []

for gene in genes_list:
    mock_data_i = np.loadtxt(f"SerializedObjects/GPSamples/MOCK_{gene}_GP_PREDICTS.csv",delimiter=",")
    rv_data_i = np.loadtxt(f"SerializedObjects/GPSamples/RV_{gene}_GP_PREDICTS.csv",delimiter=",")

    mock_data.append(mock_data_i)
    rv_data.append(rv_data_i)


# Conduct estimation procedure
NSAMPLES = mock_data[0].shape[1]
#NSAMPLES = 3

print(NSAMPLES)

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
    for j in range(len(genes_list)):
        # Select data
        mock_data_j = mock_data[j][:,i]
        rv_data_j = rv_data[j][:,i]

        # Spline interpolation
        mock_spline_j = scipy.interpolate.CubicSpline(t.flatten(),mock_data_j)
        rv_spline_j = scipy.interpolate.CubicSpline(t.flatten(),rv_data_j)

        mock_splines.append(mock_spline_j)
        rv_splines.append(rv_spline_j)

    # Aggregate observations and derivatives
    mock_preds = [mock_splines[k](t) for k in range(len(genes_list))]
    mock_derivs = [mock_splines[k].derivative(1)(t) for k in range(len(genes_list))]
    mock_preds = np.hstack(mock_preds)
    mock_derivs = np.hstack(mock_derivs)

    rv_preds = [rv_splines[k](t) for k in range(len(genes_list))]
    rv_derivs = [rv_splines[k].derivative(1)(t) for k in range(len(genes_list))]
    rv_preds = np.hstack(rv_preds)
    rv_derivs = np.hstack(rv_derivs)

    # Compute coefficients and predictions
    this_mock_coeffs = []
    this_mock_pred = []
    this_mock_soln = []

    this_rv_coeffs = []
    this_rv_pred = []
    this_rv_soln = []

    for j in range(len(genes_list)):
        mock_coeffs_j = np.linalg.pinv(mock_preds.T @ mock_preds + LAMBDA*np.eye(len(genes_list))) @ mock_preds.T @ mock_derivs[:,j]
        #mock_coeffs_j = np.linalg.pinv(mock_preds.T @ mock_preds) @ mock_preds.T @ mock_derivs[:,j]
        mock_prediction_j =  mock_preds @ mock_coeffs_j
        mock_soln_j = np.cumsum(mock_prediction_j*(t[1]-t[0]))+mock_preds[0,j]

        rv_coeffs_j = np.linalg.pinv(rv_preds.T @ rv_preds + LAMBDA*np.eye(len(genes_list))) @ rv_preds.T @ rv_derivs[:,j]
        rv_prediction_j = rv_preds @ rv_coeffs_j
        rv_soln_j = np.cumsum(rv_prediction_j*(t[1]-t[0]))+rv_preds[0,j]

        this_mock_coeffs.append(mock_coeffs_j)
        this_mock_pred.append(mock_prediction_j)
        this_mock_soln.append(mock_soln_j)

        this_rv_coeffs.append(rv_coeffs_j)
        this_rv_pred.append(rv_prediction_j)
        this_rv_soln.append(mock_soln_j)

    '''
    # These lines give us confidence that dX/dt = XA.)
    # This means that rows of A correspond to influencing parameters, not columns.
    # So, A_ij = influence of ith state on jth state
    print(mock_preds.shape)
    print(mock_coeffs_j.shape)
    print()
    print(mock_derivs[:,j])
    print()
    print(mock_preds @ mock_coeffs_j)
    print((mock_preds @ mock_coeffs_j).shape)
    print()
    print(sum(np.square(mock_derivs[:,j]-(mock_preds @ mock_coeffs_j))))
    '''

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

#print(mock_coefficients[0])
#print(rv_coefficients[0])

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