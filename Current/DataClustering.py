"""
INPUTS: 
1. gene expression data
2. pre-defined sets of genes of interest.

OUTPUTS: cluster assignments computed by dirichlet process
1. Cluster-Gene crosswalks
2. Cleaned and reformatted data for both Mock-PR8 and RV-PR8 datasets.
"""

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

gene_data = pd.read_csv("OriginalData/GeneExpressionData_MultipleSamples.tsv",sep="\t")

# Load gene sets
IFN_Alpha_Genes = list(pd.read_csv("OriginalData/IFN-Alpha_Genes",sep="\t")["Gene Symbol"])
IFN_Gamma_Genes = list(pd.read_csv("OriginalData/IFN-Gamma_Genes",sep="\t")["Gene Symbol"])

# Get list of genes we will look at.
Genes_Of_Interest = list(set(IFN_Alpha_Genes+IFN_Gamma_Genes))

# Manage Original Data
columns = list(gene_data.columns)
MOCK_PR8_columns = [s for s in columns if "mock." in s] + [s for s in columns if "mock_PR8" in s]
RV_PR8_columns = [s for s in columns if "RV." in s] + [s for s in columns if "RV_PR8" in s]

Columns_Of_Interest = list(set(MOCK_PR8_columns+RV_PR8_columns))

# Get subset of data
Data_Of_Interest = gene_data[gene_data["GeneSymbol"].isin(Genes_Of_Interest)][Columns_Of_Interest]
numpy_Data = Data_Of_Interest.to_numpy()

# Log transform to improve scale, we use eps=1e-2.
log_numpy_Data = np.log10(numpy_Data+1e-2)

# Specify the maximum number of clusters.
MAX_CLUSTERS = int(log_numpy_Data.shape[0]/20)

# Use dirichlet process (stick breaking approximation) to define the number of clusters. In practice, the method generally clusters to the top.
bgm = BayesianGaussianMixture(n_components = MAX_CLUSTERS,init_params='kmeans',random_state=42,verbose=1, max_iter=1000, tol=1e-16, weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=1e-4).fit_predict(log_numpy_Data)

unique_vals, counts = np.unique(bgm, return_counts=True)

clustered_data = gene_data[gene_data["GeneSymbol"].isin(Genes_Of_Interest)]
clustered_data["Cluster"]=bgm

# Save cluster cross-walk.
Cluster_Crosswalk = clustered_data[["GeneSymbol","Cluster"]]
Cluster_Crosswalk.to_csv("ProcessedData/Cluster_Crosswalk.csv",index=False)

# Now we want to more clearly separate data into the Mock and RV groups.
MOCK_PR8_columns = ["GeneSymbol"] + [s for s in columns if "mock." in s] + [s for s in columns if "mock_PR8" in s]
RV_PR8_columns = ["GeneSymbol"] + [s for s in columns if "RV." in s] + [s for s in columns if "RV_PR8" in s]

CLUSTERED_MOCK_PR8 = clustered_data[MOCK_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
CLUSTERED_RV_PR8 = clustered_data[RV_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()

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

CLUSTERED_MOCK_PR8_map = make_index_map(CLUSTERED_MOCK_PR8.index)
CLUSTERED_MOCK_PR8.rename(index=CLUSTERED_MOCK_PR8_map, inplace=True)

CLUSTERED_RV_PR8_map = make_index_map(CLUSTERED_RV_PR8.index)
CLUSTERED_RV_PR8.rename(index=CLUSTERED_RV_PR8_map, inplace=True)

CLUSTERED_MOCK_PR8.to_csv("ProcessedData/CLUSTERED_MOCK_PR8.csv",sep=",",index_label="Day")
CLUSTERED_RV_PR8.to_csv("ProcessedData/CLUSTERED_RV_PR8.csv",sep=",",index_label="Day")