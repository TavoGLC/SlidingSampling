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
##############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

from itertools import product
from collections import Counter
from numpy import linalg as LA

import matplotlib.pyplot as plt

###############################################################################
# Global definitions
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  

seqDataDir=SequencesDir+'/sequences.fasta'

sequencesFrags = SequencesDir + '/splitted'

matrixData = GlobalDirectory + '/frames'

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

###############################################################################
# Sequence Graphs Functions
###############################################################################

def MakeAdjacencyList(processedSequence,Block,skip=0):
    
    CharactersToLocation = dict([(val,k) for k,val in enumerate(Block)])
    
    x,y = [],[]
    
    for k in range(len(processedSequence)-skip-1):
        backFragment = processedSequence[k]
        forwardFragment = processedSequence[k+skip+1]
            
        if backFragment in Block and forwardFragment in Block:
            x.append(CharactersToLocation[backFragment])
            y.append(CharactersToLocation[forwardFragment])
            
    return x,y

def RelationalSkip(Sequence,Block,skip=0):
    
    D12 = np.zeros((len(Block),len(Block)))
    currentMatrix = np.zeros((len(Block),len(Block)))
    fragmentSize=len(Block[0])
    
    processedSequence=SplitString(Sequence,fragmentSize)
    x,y = MakeAdjacencyList(processedSequence,Block,skip=skip)
    
    pairs = [val for val in zip(x,y)]
    counts = Counter(pairs)
    
    for ky in counts.keys():
        currentMatrix[ky] = counts[ky]
        
    currentMatrix = currentMatrix + currentMatrix.T
    
    
    for k,val in enumerate(currentMatrix.sum(axis=0)):
        D12[k,k] = 1/np.sqrt(2*val)
    
    currentMatrix = np.dot(D12,currentMatrix).dot(D12)
    w,v = LA.eig(currentMatrix)
    norm = LA.norm(w)
    
    return currentMatrix/norm

def MakeBidirectionalEncoding(Sequence,Block,skip):
    
    relationalForward = RelationalSkip(Sequence,Block,skip=skip)
    relationalForward = (relationalForward-relationalForward.min())/(relationalForward.max()-relationalForward.min())
    
    relationalReverse = RelationalSkip(str(Sequence)[::-1],Block,skip=skip)
    relationalReverse = (relationalReverse-relationalReverse.min())/(relationalReverse.max()-relationalReverse.min())
    
    return relationalForward,relationalReverse


def Make3DBidirectionalEncoding(Sequence,Block):
    
    matrix = np.zeros((len(Block),len(Block),2))
    
    localMatrix = MakeBidirectionalEncoding(Sequence,Block,skip=1)
    matrix[:,:,0] = localMatrix[0]
    matrix[:,:,1] = localMatrix[1]
        
    return matrix

def Make4DEncoding(Sequence,Block=Blocks[1],fragments=16):
    
    Sequence = Sequence.seq
    step = len(Sequence)//fragments
    container = np.zeros(shape=(fragments,len(Block),len(Block),2))
    toStringSequence = str(Sequence)
    
    for k in range(fragments):
        localSequence = toStringSequence[k*step:(k+1)*step]
        currentEncoding = Make3DBidirectionalEncoding(localSequence,Block)
        container[k,:,:,:] = currentEncoding 
        
    return container 

def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequence Graphs Functions
###############################################################################

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]
fragmentsDirs = np.array(fragmentsDirs).reshape(-1,11)

for blk in fragmentsDirs:
    Container = []
    for val in blk:
        Container = Container + GetSeqs(val)
    
    print(len(Container))

    names = [seq.id for seq in Container]
    data = GetDataParallel(Container,Make4DEncoding)
    
    for nme, db in zip(names,data):
        np.save(matrixData+'/'+nme, db)
