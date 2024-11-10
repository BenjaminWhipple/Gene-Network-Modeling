"""
ABOUT:


INPUTS:
    

OUTPUTS:


"""

import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import dill as pickle

"""
TODO:
1. Plot parameter distributions
2. Plot parameter differences histogram.
"""

# Determine Global Params
CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))
mock_data_0 = np.loadtxt(f"SerializedObjects/ClusterGPPredicts/MOCK_PR8_CLUSTER_0_GP_PREDICTS.csv",delimiter=",")
NSAMPLES = mock_data_0.shape[1]

# Load parameters
with open("SerializedObjects/EstimationResults/MOCK_PR8_COEFFICIENTS.pkl",'rb') as f:
    mock_coefficients = pickle.load(f)
with open("SerializedObjects/EstimationResults/RV_PR8_COEFFICIENTS.pkl",'rb') as f:
    rv_coefficients = pickle.load(f)

nrow,ncol = mock_coefficients[0].shape

# Create Parameter Distributions Figures
# This works, but is not very clear.

fig, axs = plt.subplots(10,10,figsize=(2.5*nrow,2*ncol),constrained_layout=True)
for i in range(nrow):
	for j in range(ncol):
		coeffs = [mock_coefficients[k][i,j] for k in range(NSAMPLES)]
		axs[i,j].hist(coeffs, density = True)

fig.savefig("Images/DiagnosticImages/FittedParameterDistributions/Mock_Parameters.png")

fig, axs = plt.subplots(10,10,figsize=(2.5*nrow,2*ncol),constrained_layout=True)
for i in range(nrow):
	for j in range(ncol):
		coeffs = [rv_coefficients[k][i,j] for k in range(NSAMPLES)]
		axs[i,j].hist(coeffs, density = True)

fig.savefig("Images/DiagnosticImages/FittedParameterDistributions/RV_Parameters.png")


# Now, plot heatmaps of the coefficients.

mock_coeffs = np.array(mock_coefficients)
rv_coeffs = np.array(rv_coefficients)

differences = rv_coeffs - mock_coeffs  # Shape will be (1000, 10, 10)

median_diff = np.median(differences, axis=0)  # Shape (10, 10) - median across samples
ci_lower = np.percentile(differences, 2.5, axis=0)  # Shape (10, 10) - 2.5th percentile
ci_upper = np.percentile(differences, 97.5, axis=0)  # Shape (10, 10) - 97.5th percentile

# Results:
print("Median of Differences:\n", median_diff)
print("\n95% CI Lower Bound:\n", ci_lower)
print("\n95% CI Upper Bound:\n", ci_upper)

# Determine the color scale limits based on the min and max across all heatmaps
vmin = min(ci_lower.min(), median_diff.min(), ci_upper.min())
vmax = max(ci_lower.max(), median_diff.max(), ci_upper.max())

# Set up the figure and axes for the 1x3 layout
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define heatmap settings with shared vmin and vmax
heatmap_settings = {
    "annot": True,
    "fmt": ".2f",
    "xticklabels": np.arange(10),
    "yticklabels": np.arange(10),
    #"cbar_kws": {"label": "Difference Value"},
    "vmin": vmin,
    "vmax": vmax,
    "annot_kws": {"size": 6}
}

# Plot the 2.5th percentile heatmap
sns.heatmap(ci_lower, ax=axes[0], cmap='coolwarm', **heatmap_settings)
axes[0].set_title("2.5th Percentile of Differences")
axes[0].set_xlabel("Cluster (Influenced by)")
axes[0].set_ylabel("Cluster (Influenced)")

# Plot the median heatmap
sns.heatmap(median_diff, ax=axes[1], cmap='coolwarm', **heatmap_settings)
axes[1].set_title("Median of Differences")
axes[1].set_xlabel("Cluster (Influenced by)")
#axes[1].set_ylabel("Row")

# Plot the 97.5th percentile heatmap
sns.heatmap(ci_upper, ax=axes[2], cmap='coolwarm', **heatmap_settings)
axes[2].set_title("97.5th Percentile of Differences")
axes[2].set_xlabel("Cluster (Influenced by)")
#axes[2].set_ylabel("Row")

# Adjust layout to ensure proper spacing
plt.tight_layout()
plt.savefig("Images/DiagnosticImages/DifferencesHeatmap.png")