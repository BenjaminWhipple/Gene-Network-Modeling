"""
INPUTS:
	Gene to Cluster crosswalk: "ProcessedData/Cluster_Crosswalk.csv"
	Formatted Gene Data for MOCK-PR8 group: "ProcessedData/CLUSTERED_MOCK_PR8.csv"
	Formatted Gene Data for RV-PR8 group: "ProcessedData/CLUSTERED_RV_PR8.csv"

OUTPUTS:
	Dataframes for each cluster between RV-PR8 and MOCK-PR8 groups: "ProcessedData/ClusterMeasurements/*.csv" for general *.
"""

import sklearn
import pandas as pd
import numpy as np

ClusterCrosswalk = pd.read_csv("ProcessedData/Cluster_Crosswalk.csv")
MOCK_PR8_DATA = pd.read_csv("ProcessedData/CLUSTERED_MOCK_PR8.csv")
RV_PR8_DATA = pd.read_csv("ProcessedData/CLUSTERED_RV_PR8.csv")

Clusters = np.unique(ClusterCrosswalk["Cluster"])

GenesByCluster = []

# FOR REFACTOR: MERGE FOLLOWING LOOPS.

# Create cluster number, gene list pairs.
for i in Clusters:
	temp = ClusterCrosswalk[ClusterCrosswalk["Cluster"]==i]
	GenesByCluster.append((int(i),list(temp["GeneSymbol"])))

# Create cluster measurements from previous pairing.
for group in GenesByCluster:
	idx, Genes = group
	MOCK_PR8_TEMP = []
	RV_PR8_TEMP = []
	for gene in Genes:
		temp1 = MOCK_PR8_DATA.loc[:,["Day",gene]].to_numpy()
		temp2 = RV_PR8_DATA.loc[:,["Day",gene]].to_numpy()

		MOCK_PR8_TEMP.append(temp1)
		RV_PR8_TEMP.append(temp2)

	temp_df = pd.DataFrame(data=np.vstack(MOCK_PR8_TEMP),columns=["Day",f"Cluster_{idx}_Expression"])
	temp_df.to_csv(f"ProcessedData/ClusterMeasurements/MOCK_PR8_Cluster_{idx}_Measurements.csv")

	temp_df = pd.DataFrame(data=np.vstack(RV_PR8_TEMP),columns=["Day",f"Cluster_{idx}_Expression"])
	temp_df.to_csv(f"ProcessedData/ClusterMeasurements/RV_PR8_Cluster_{idx}_Measurements.csv")