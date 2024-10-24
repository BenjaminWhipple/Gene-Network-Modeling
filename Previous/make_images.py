#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:01:14 2024

@author: benjaminwhipple
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

GenesOfInterest = list(pd.read_csv("OriginalData/NewGenesOfInterest.csv",sep=",")["Gene Symbol"])

MOCK_DATA = pd.read_csv("ProcessedData/Mock-PR8").rename(columns={"Unnamed: 0": "Day"})
RV_DATA = pd.read_csv("ProcessedData/RV-PR8").rename(columns={"Unnamed: 0": "Day"})

times = np.loadtxt("SurrogateSamples/SAMPLE_TIMES.txt",delimiter=" ")

Groups = ["Mock-PR8","RV-PR8"]

Mock_Data = {}
RV_Data = {}

for gene in GenesOfInterest:
    Mock_Data[gene]=np.loadtxt(f"SurrogateSamples/Mock-PR8_{gene}.txt",delimiter=",")
    RV_Data[gene]=np.loadtxt(f"SurrogateSamples/RV-PR8_{gene}.txt",delimiter=" ")
    #print(Mock_Data[gene][:,1].shape)
    #print(np.percentile(Mock_Data[gene],97.5,axis=1).shape)

for gene in GenesOfInterest:
    fig, axs = plt.subplots(1,2,sharex=True,sharey=True,squeeze=True,figsize=(8,4))
    fig.suptitle(f"{gene}",fontsize=14)
    axs[0].set_title("RV-PR8")
    #axs[0].plot(times, RV_Data[gene],alpha=0.1,color="cornflowerblue")
    
    rv_upper = np.percentile(RV_Data[gene],97.5,axis=1)
    rv_lower = np.percentile(RV_Data[gene],2.5,axis=1)
    #axs[0].plot(times,rv_upper,color="black")
    axs[0].fill_between(times,rv_lower,rv_upper,color="cornflowerblue",alpha=0.75)
    
    axs[0].plot(times, np.mean(RV_Data[gene],axis=1),color="black")
    axs[0].plot(RV_DATA["Day"],np.log10(RV_DATA[gene]),"o",color="tomato")
    axs[0].set_xticks([0,1,2,3,4,5,6])
    #axs[0].set_yticks([1,2,3,4,5,6])
    axs[0].set_xlabel("Days Post PR8 Infection")
    axs[0].set_ylabel("Expression (log10 read counts)")
    
    axs[1].set_title("Mock-PR8")
    #axs[1].plot(times, Mock_Data[gene],alpha=0.1,color="cornflowerblue")
    mock_upper = np.percentile(Mock_Data[gene],97.5,axis=1)
    mock_lower = np.percentile(Mock_Data[gene],2.5,axis=1)
    #axs[0].plot(times,rv_upper,color="black")
    axs[1].fill_between(times,mock_lower,mock_upper,color="cornflowerblue",alpha=0.75)
    axs[1].plot(times, np.mean(Mock_Data[gene],axis=1),color="black")
    axs[1].plot(MOCK_DATA["Day"],np.log10(MOCK_DATA[gene]),"o",color="tomato")
    axs[1].set_xlabel("Days Post PR8 Infection")

    plt.savefig(f"Images/GeneComparisons/{gene}_comparison.png",dpi=300)
    #axs[1].set_xticks([1.0,2.0,3.0,4.0,5.0,6.0])
    

    pass

#[f"{Group}_{gene}.txt" for gene in ]