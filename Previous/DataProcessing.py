"""
Note: This file is not currently very easily extendable.
"""

import pandas as pd
gene_data = pd.read_csv("OriginalData/OldGeneExpressionData_MultipleSamples.tsv",sep="\t")

IFN_Alpha_Genes = list(pd.read_csv("OriginalData/IFN-Alpha_Genes",sep="\t")["Gene Symbol"])
IFN_Gamma_Genes = list(pd.read_csv("OriginalData/IFN-Gamma_Genes",sep="\t")["Gene Symbol"])

# We want to reorganize the data into groups as files, genes as columns, days as rows.
columns = list(gene_data.columns)

MOCK_PR8_columns = ["GeneSymbol"] + [s for s in columns if "mock." in s] + [s for s in columns if "mock_PR8" in s]
RV_PR8_columns = ["GeneSymbol"] + [s for s in columns if "RV." in s] + [s for s in columns if "RV_PR8" in s]

IFN_Alpha_Gene_Data = gene_data[gene_data["GeneSymbol"].isin(IFN_Alpha_Genes)]
IFN_Gamma_Gene_Data = gene_data[gene_data["GeneSymbol"].isin(IFN_Gamma_Genes)]

IFN_Alpha_MOCK_PR8 = IFN_Alpha_Gene_Data[MOCK_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
IFN_Alpha_RV_PR8 = IFN_Alpha_Gene_Data[RV_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
IFN_Gamma_MOCK_PR8 = IFN_Gamma_Gene_Data[MOCK_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()
IFN_Gamma_RV_PR8 = IFN_Gamma_Gene_Data[RV_PR8_columns].copy().rename(columns={"GeneSymbol":"Day"}).set_index("Day").transpose()

# This function is highly specific to our data.
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

IFN_Alpha_MOCK_PR8_map = make_index_map(IFN_Alpha_MOCK_PR8.index)
IFN_Alpha_MOCK_PR8.rename(index=IFN_Alpha_MOCK_PR8_map, inplace=True)

IFN_Alpha_RV_PR8_map = make_index_map(IFN_Alpha_RV_PR8.index)
IFN_Alpha_RV_PR8.rename(index=IFN_Alpha_RV_PR8_map, inplace=True)

IFN_Gamma_MOCK_PR8_map = make_index_map(IFN_Gamma_MOCK_PR8.index)
IFN_Gamma_MOCK_PR8.rename(index=IFN_Gamma_MOCK_PR8_map, inplace=True)

IFN_Gamma_RV_PR8_map = make_index_map(IFN_Gamma_RV_PR8.index)
IFN_Gamma_RV_PR8.rename(index=IFN_Gamma_RV_PR8_map, inplace=True)

IFN_Alpha_MOCK_PR8.to_csv("ProcessedData/IFN-Alpha_Mock-PR8.csv")
IFN_Alpha_RV_PR8.to_csv("ProcessedData/IFN-Alpha_RV-PR8")
IFN_Gamma_MOCK_PR8.to_csv("ProcessedData/IFN-Gamma_Mock-PR8")
IFN_Gamma_RV_PR8.to_csv("ProcessedData/IFN-Gamma_RV-PR8")
