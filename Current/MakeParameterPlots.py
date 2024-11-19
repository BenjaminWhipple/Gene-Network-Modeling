import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import dill as pickle

# Load Genes
genes_list = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv")["Gene Symbol"].str.lower())

print(genes_list)

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

# Load parameters
with open("SerializedObjects/EstimationResults/MOCK_PR8_COEFFICIENTS.pkl",'rb') as f:
    mock_coefficients = pickle.load(f)
with open("SerializedObjects/EstimationResults/RV_PR8_COEFFICIENTS.pkl",'rb') as f:
    rv_coefficients = pickle.load(f)

nrow,ncol = mock_coefficients[0].shape

# These plots are not useful for moderate to large numbers of genes.
"""
print("Here")
# Create Parameter Distributions Figures
# This works, but is not very clear.

fig, axs = plt.subplots(nrow,ncol,figsize=(2.5*nrow,2*ncol),constrained_layout=True)
for i in range(nrow):
	for j in range(ncol):
		coeffs = [mock_coefficients[k][i,j] for k in range(NSAMPLES)]
		axs[i,j].hist(coeffs, density = True)

print("Here")

fig.savefig("Images/DiagnosticImages/FittedParameterDistributions/Mock_Parameters.png")

print("Here")

fig, axs = plt.subplots(nrow,ncol,figsize=(2.5*nrow,2*ncol),constrained_layout=True)
for i in range(nrow):
	for j in range(ncol):
		coeffs = [rv_coefficients[k][i,j] for k in range(NSAMPLES)]
		axs[i,j].hist(coeffs, density = True)

print("Here")


fig.savefig("Images/DiagnosticImages/FittedParameterDistributions/RV_Parameters.png")
"""

print("Here")

# Now, plot heatmaps of the coefficients.

mock_coeffs = np.array(mock_coefficients)
rv_coeffs = np.array(rv_coefficients)

#print(np.median(mock_coeffs,axis=0).shape)

differences = rv_coeffs - mock_coeffs  # Shape will be (1000, 10, 10)

median_diff = np.median(differences, axis=0)  # Shape (10, 10) - median across samples
ci_lower = np.percentile(differences, 2.5, axis=0)  # Shape (10, 10) - 2.5th percentile
ci_upper = np.percentile(differences, 97.5, axis=0)  # Shape (10, 10) - 97.5th percentile


median_mock = np.median(mock_coeffs,axis=0)
median_rv = np.median(rv_coeffs,axis=0)

np.savetxt("SerializedObjects/median_mock_coeffs.csv",median_mock, delimiter=",")
np.savetxt("SerializedObjects/median_rv_coeffs.csv",median_rv, delimiter=",")
np.savetxt("SerializedObjects/median_diff_coeffs.csv",median_diff, delimiter=",")
# Results:
#print("Median of Differences:\n", median_diff)
#print("\n95% CI Lower Bound:\n", ci_lower)
#print("\n95% CI Upper Bound:\n", ci_upper)

# Determine the color scale limits based on the min and max across all heatmaps
vmin = min(ci_lower.min(), median_diff.min(), ci_upper.min())
vmax = max(ci_lower.max(), median_diff.max(), ci_upper.max())

# Set up the figure and axes for the 1x3 layout
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define heatmap settings with shared vmin and vmax
heatmap_settings = {
    "annot": True,
    "fmt": ".2f",
    "xticklabels": genes_list,
    "yticklabels": genes_list,
    #"cbar_kws": {"label": "Difference Value"},
    "vmin": vmin,
    "vmax": vmax,
    "annot_kws": {"size": 6}
}

heatmap_settings2 = {
    "annot": False,
    "fmt": ".2f",
    "xticklabels": genes_list,
    "yticklabels": genes_list,
    #"cbar_kws": {"label": "Difference Value"},
    "vmin": median_diff.min(),
    "vmax": median_diff.max(),
    "annot_kws": {"size": 6}
}

# Plot the 2.5th percentile heatmap
sns.heatmap(ci_lower, ax=axes[0], cmap='coolwarm', **heatmap_settings)
axes[0].set_title("2.5th Percentile of Differences")
axes[0].set_xlabel("Cluster (Influenced)")
axes[0].set_ylabel("Cluster (Influencer)")

# Plot the median heatmap
sns.heatmap(median_diff, ax=axes[1], cmap='coolwarm', **heatmap_settings)
axes[1].set_title("Median of Differences")
axes[1].set_xlabel("Cluster (Influenced)")
#axes[1].set_ylabel("Row")

# Plot the 97.5th percentile heatmap
sns.heatmap(ci_upper, ax=axes[2], cmap='coolwarm', **heatmap_settings)
axes[2].set_title("97.5th Percentile of Differences")
axes[2].set_xlabel("Cluster (Influenced)")
#axes[2].set_ylabel("Row")

# Adjust layout to ensure proper spacing
plt.tight_layout()
plt.savefig("Images/DiagnosticImages/DifferencesHeatmap.png")

plt.figure(figsize=(10,10))
sns.heatmap(median_diff, cmap='coolwarm', **heatmap_settings2)
plt.title("Median of Differences (RV-PR8 - Mock-PR8)")
plt.xlabel("Gene (Influenced)")
plt.ylabel("Gene (Influencer)")
plt.tight_layout()
plt.savefig("Images/DiagnosticImages/MedianDifferencesHeatmap.png",dpi=300)

plt.figure(figsize=(10,10))
sns.heatmap(np.median(mock_coeffs,axis=0), cmap='coolwarm', **heatmap_settings2)
plt.title("Mock-PR8 Median Coefficients")
plt.xlabel("Gene (Influenced)")
plt.ylabel("Gene (Influencer)")
plt.tight_layout()
plt.savefig("Images/DiagnosticImages/MockCoeffsHeatmap.png",dpi=300)

plt.figure(figsize=(10,10))
sns.heatmap(np.median(rv_coeffs,axis=0), cmap='coolwarm', **heatmap_settings2)
plt.title("RV-PR8 Median Coefficients")
plt.xlabel("Gene (Influenced)")
plt.ylabel("Gene (Influencer)")
plt.tight_layout()
plt.savefig("Images/DiagnosticImages/RVCoeffsHeatmap.png",dpi=300)