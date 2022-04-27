#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 00:04:53 2022

@author: tavoglc
"""

###############################################################################
# Loading packages 
###############################################################################

import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.spatial import distance as ds
from sklearn.manifold import SpectralEmbedding

###############################################################################
# Visualization functions
###############################################################################

def PlotStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)
    
###############################################################################
# Data cleaning
###############################################################################
    
with open(r'/media/tavoglc/storage/backup/main2/main/mining/databases/textdata.txt') as f:
    lines = f.readlines()

divisionPoints = []

for k,val in enumerate(lines):
    if val[0:2]=='ID':
        divisionPoints.append(k)
    
divisionPoints.append(len(lines))

dataPerId = []

for k in range(len(divisionPoints)-1):
    dataPerId.append(lines[divisionPoints[k]:divisionPoints[k+1]])

accesions = [val[1].split()[1][0:-1] for val in dataPerId]

gnnames = []

for val in dataPerId:
    for sal in val:
        if sal[0:2]=='GN':
            gnnames.append(sal.split()[1])
            break

gnames = [val[val.find('=')+1::] for val in gnnames]

termsPerId = []

for val in dataPerId:
    innerContainer = []
    for sal in val:
        if sal[0:2]=='DR':
            if re.match('(.*)GO(.*)',sal) or re.match('(.*)InterPro(.*)',sal) or re.match('(.*)Pfam(.*)',sal):
                innerContainer.append(sal[5:-2])
    termsPerId.append(innerContainer)
    
uniqueTerms = list(set([item for sublist in termsPerId for item in sublist]))

termToInt = dict([(val,k) for k,val in enumerate(uniqueTerms)])
intToTerm = dict([(k,val) for k,val in enumerate(uniqueTerms)])

GOindex = [k for k, val in enumerate(uniqueTerms) if val[0:2]=='GO']
InterIndex = [k for k,val in enumerate(uniqueTerms) if re.match('(.*)InterPro(.*)',val)]
PfamIndex = [k for k,val in enumerate(uniqueTerms) if re.match('(.*)Pfam(.*)',val)]

GoDescriptions = [' '.join(uniqueTerms[k].split()[2:-1]) for k in GOindex]
intToGoDescription = dict([(k,val) for k,val in enumerate(GoDescriptions)])

InterDescriptions = [uniqueTerms[k].split()[2] for k in InterIndex]
intToInterDescription = dict([(k,val) for k,val in enumerate(InterDescriptions)])

PfamDescriptions = [uniqueTerms[k].split()[2] for k in PfamIndex]
intToPfamDescription = dict([(k,val) for k,val in enumerate(PfamDescriptions)])

###############################################################################
# Dataset generation
###############################################################################

termsData = np.zeros((len(dataPerId),len(uniqueTerms)))

for k,val in enumerate(termsPerId):
    for trm in val:
        loc = termToInt[trm]
        termsData[k,loc] = termsData[k,loc] + 1
        
###############################################################################
# Dimensionality reduction
###############################################################################

distances = ds.pdist(termsData,'russellrao')
distances = ds.squareform(distances)

embedding = SpectralEmbedding(n_components=2)

emb = embedding.fit_transform(distances)
emb = (emb - emb.min())/(emb.max()-emb.min())

###############################################################################
# Clustering 
###############################################################################

clusters = DBSCAN(eps=0.035, min_samples=4).fit(emb)
labels = clusters.labels_

def MakeClusterPlot(EmbData,Labels,Data,index,termdict):

    uniqueLabels = np.unique(Labels)
    colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(uniqueLabels))]
    
    plt.figure(figsize=(12,10))
    for k,lab in enumerate(uniqueLabels):
        
        msk = [k for k,val in enumerate(Labels) if val==lab]
        localData = Data[msk,:]
        localData = localData[:,index]
        order = np.argsort(localData.sum(axis=0))[::-1][0:3]
        terms = [termdict[val] for val in order]
        
        location = EmbData[msk,:].mean(axis=0)
        
        if lab==-1:
            colr = [0,0,0,1]
        else:
            colr = colors[k]
        
        plt.plot(EmbData[msk,0],EmbData[msk,1],'o',c=colr,label = 'Size = '+str(len(msk)) )
        
        for k,val in enumerate(terms):
            plt.text(location[0]+0.05,location[1]-k*0.03,val,fontsize=8)

    plt.legend(bbox_to_anchor=(1.0, 1.0))
    ax = plt.gca()
    PlotStyle(ax)


counter = 0
DPI = 300

MakeClusterPlot(emb,labels,termsData,GOindex,intToGoDescription)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakeClusterPlot(emb,labels,termsData,InterIndex,intToInterDescription)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakeClusterPlot(emb,labels,termsData,PfamIndex,intToPfamDescription)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1


