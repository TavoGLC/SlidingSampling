#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2021 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo
"""

###############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import networkx as nx
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

from itertools import product
from numpy import linalg as LA

###############################################################################
# Global definitions
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  

seqDataDir=SequencesDir+'/sequences.fasta'

sequencesFrags = SequencesDir + '/splitted'

matrixData = GlobalDirectory + '/featuresdata'

MaxCPUCount=int(0.90*mp.cpu_count())

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 4
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

##############################################################################
# Sequence Loading functions
###############################################################################

#Wrapper function to load the sequences
def GetSeqs(Dir):
    
    cDir=Dir
    
    with open(cDir) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList  

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def SplitString(String,ChunkSize):
    '''
    Split a string ChunkSize fragments using a sliding windiow

    Parameters
    ----------
    String : string
        String to be splitted.
    ChunkSize : int
        Size of the fragment taken from the string .

    Returns
    -------
    Splitted : list
        Fragments of the string.

    '''
    try:
        localString=str(String.seq)
    except AttributeError:
        localString=str(String)
      
    if ChunkSize==1:
        Splitted=[val for val in localString]
    
    else:
        nCharacters=len(String)
        Splitted=[localString[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    '''
    Creates a dictionary that takes a Unique element as key and return its 
    position in the UniqueElements array
    Parameters
    ----------
    UniqueElements : List,array
        list of unique elements.

    Returns
    -------
    localDictionary : dictionary
        Maps element to location.

    '''
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

###############################################################################
# Sequences as graphs. 
###############################################################################

def MakeSequenceGraph(Sequence,NodeNames,scheme='A',viz=False):
    
    Nodes=np.arange(len(NodeNames))
    localDict=UniqueToDictionary(NodeNames)
    
    if viz:
        localGraph=nx.Graph()
    else:    
        localGraph=nx.MultiGraph()
        
    localGraph.add_nodes_from(Nodes)
    
    if scheme == 'A':
        
        fragmentSize=len(NodeNames[0])
        processedSequence=SplitString(Sequence,fragmentSize)
        
        for k in range(len(processedSequence)-1):
            
            if processedSequence[k] in NodeNames and processedSequence[k+1] in NodeNames:
            
                current=localDict[processedSequence[k]]
                forward=localDict[processedSequence[k+1]]
                localGraph.add_edge(current,forward)
    
    elif scheme == 'B':
        
        fragmentSize=2*len(NodeNames[0])
        processedSequence=SplitString(Sequence,fragmentSize)
        
        for frag in processedSequence:
            
            backFragment = frag[0:int(fragmentSize/2)] 
            forwardFragment = frag[int(fragmentSize/2)::]
            
            if backFragment in NodeNames and forwardFragment in NodeNames:
                
                current = localDict[backFragment]
                forward = localDict[forwardFragment]
                localGraph.add_edge(current,forward)
    
    return localGraph

###############################################################################
# Sequences as graphs. 
###############################################################################

def MakeNormAdjacencyMatrix(graph):
    
    matrixShape = (len(graph.nodes),len(graph.nodes))
    D12 = np.zeros(matrixShape)
    
    for (node, val) in graph.degree():
        D12[node,node] = 1/np.sqrt(val)
        
    A = nx.adjacency_matrix(graph).toarray()    
    normA = np.dot(D12,A).dot(D12)
    
    w,v = LA.eig(normA)
        
    return normA/LA.norm(w)

def MakeSequenceMatrix(sequence,blocks=Blocks):
    
    container = []
    mat = np.zeros((64,80))
    
    for blk in blocks:
        graphA = MakeSequenceGraph(sequence,blk)
        graphB = MakeSequenceGraph(sequence,blk,scheme='B')
    
        a = MakeNormAdjacencyMatrix(graphA)
        b = MakeNormAdjacencyMatrix(graphB)
    
        c = a-b
        c = (c-c.min())/(c.max()-c.min())
        container.append(c)
    
    mat[0:4,0:4] = container[0]
    mat[4:20,0:16] = container[1]
    mat[0:64,16:80] = container[2]
    
    return mat

def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequences as graphs. 
###############################################################################

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]
fragmentsDirs = np.array(fragmentsDirs).reshape(-1,11)

for blk in fragmentsDirs:
    Container = []
    for val in blk:
        Container = Container +GetSeqs(val)
    
    print(len(Container))
    names = [seq.id for seq in Container]
    data = GetDataParallel(Container,MakeSequenceMatrix)
    
    for nme, db in zip(names,data):
        np.save(matrixData+'/'+nme, db)
