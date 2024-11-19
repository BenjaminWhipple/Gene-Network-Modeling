# Compute Eigenvector centrality.

# This following should be moved to a different file.
import networkx as nx
import numpy as np
import pandas as pd
import csv

genes_list = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv")["Gene Symbol"].str.lower())

print(genes_list)

median_mock = np.loadtxt("SerializedObjects/median_mock_coeffs.csv", delimiter=",")
median_rv = np.loadtxt("SerializedObjects/median_rv_coeffs.csv", delimiter=",")
test = nx.from_numpy_array(median_mock,create_using=nx.DiGraph)
#print(test)

labels = {i: genes_list[i] for i in range(len(genes_list))}

test = nx.relabel_nodes(test, labels)

res = nx.eigenvector_centrality_numpy(test,weight='weight')
#print(res)

#print()
sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: abs(item[1]), reverse=True)}
print(sorted_res)

with open("SerializedObjects/Mock_EigenvectorCentrality.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Gene', 'Centrality'])
    # Write each key-value pair
    for key, value in sorted_res.items():
        writer.writerow([key, value])
#mock_res = pd.DataFrame(data=sorted_res,columns = ["Genes","Mock Centrality"])

test = nx.from_numpy_array(median_rv,create_using=nx.DiGraph)
#print(test)

labels = {i: genes_list[i] for i in range(len(genes_list))}

test = nx.relabel_nodes(test, labels)

res = nx.eigenvector_centrality_numpy(test,weight='weight')
#print(res)

#print()
sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: abs(item[1]), reverse=True)}
print(sorted_res)

with open("SerializedObjects/RV_EigenvectorCentrality.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Gene', 'Centrality'])
    # Write each key-value pair
    for key, value in sorted_res.items():
        writer.writerow([key, value])

#rv_res = pd.DataFrame(data=sorted_res,columns = ["Genes","RV Centrality"])