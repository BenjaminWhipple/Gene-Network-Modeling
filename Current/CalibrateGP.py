"""
INPUTS:
	Dataframe for cluster crosswalk: "ProcessedData/Cluster_Crosswalk.csv"
	Dataframes for each cluster between RV-PR8 and MOCK-PR8 groups: "ProcessedData/ClusterMeasurements/*.csv" for general *.

OUTPUTS:
	GP models (pickled files): "SerializedObjects/ClusterGPs/*.pkl" for general *
	Spline representation of GP mean: "SerializedObjects/MeanSplines/*.pkl" for general *
"""

"""
TODO:
[]
[]
[]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# Load cluster crosswalk
CROSSWALK = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
# Determine number of clusters
CLUSTERS = np.sort(np.unique(CROSSWALK["Cluster"]))
print(CLUSTERS)

# Load ClusterMeasurements
MOCK_PR8_CLUSTERS = []
RV_PR8_CLUSTER = []

for i in CLUSTERS:
	