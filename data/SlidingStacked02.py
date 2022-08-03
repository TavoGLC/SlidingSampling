#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 

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
import pandas as pd
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

GlobalDirectory = r"/media/tavoglc/storage/data/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  

seqDataDir=SequencesDir+'/sequences.fasta'

sequencesFrags = SequencesDir + '/splitted'

matrixData = GlobalDirectory + '/stacked02'

MaxCPUCount=int(0.75*mp.cpu_count())

MetaDataPath = r"/media/tavoglc/storage/data/nov2021"
MetaDataP = MetaDataPath + '/newMetaData.csv'

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

def CountUniqueElements(UniqueElements,String,Processed=False):
    '''
    Calculates the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    Parameters
    ----------
    UniqueElements : array,list
        Elements to be analized.
    String : strting
        Sequence data.
    Processed : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    localCounter : array
        Normalized frequency of each unique fragment.
    '''
    
    nUnique = len(UniqueElements)
    localCounter = [0 for k in range(nUnique)]
    
    if Processed:
        ProcessedString = String
    else:
        ProcessedString = SplitString(String,len(UniqueElements[0]))
        
    nSeq = len(ProcessedString)
    UniqueDictionary = UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        
        if val in UniqueElements:
            
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
            
    localCounter = [val/nSeq for val in localCounter]
    localCounter = np.array(localCounter)
    localCounter = (localCounter-localCounter.min())/(localCounter.max()-localCounter.min())
    
    return localCounter

###############################################################################
# Sequence Graphs Functions
###############################################################################

def MakeAdjacencyList(processedSequence,Block,skip=0):
    '''
    Parameters
    ----------
    processedSequence : array-like, list
        Sequence sliced with a sliding window.
    Block : array-like
        Contains unique tokens to create the graph.
    skip : int, optional
        Overlap between tokens. The default is 0.

    Returns
    -------
    x : list
        contains the token number in the sequnce, part of the adjacency list.
    y : list
        contains the token number in the sequnce, part of the adjacency list.

    '''
    CharactersToLocation = dict([(val,k) for k,val in enumerate(Block)])
    
    x,y = [],[]
    
    for k in range(len(processedSequence)-skip-1):
        backFragment = processedSequence[k]
        forwardFragment = processedSequence[k+skip+1]
            
        if backFragment in Block and forwardFragment in Block:
            x.append(CharactersToLocation[backFragment])
            y.append(CharactersToLocation[forwardFragment])
            
    return x,y

def MakeSequenceEncoding(Sequence,Block):
    
    '''
    Parameters
    ----------
    Sequence : string, biopython seq object
        Sequence to analyze.
    Block : array-like,list
        Contains the unique elements to make the graph.
    Returns
    -------
    array
        normalized adjacency matrix.
    '''
    
    D12 = np.zeros((len(Block),len(Block)))
    currentMatrix = np.zeros((len(Block),len(Block)))
    fragmentSize=len(Block[0])
    skip = fragmentSize-1
    
    processedSequence=SplitString(Sequence,fragmentSize)
    x,y = MakeAdjacencyList(processedSequence,Block,skip=skip)
    
    pairs = [val for val in zip(x,y)]
    counts = Counter(pairs)
    
    for ky in counts.keys():
        currentMatrix[ky] = counts[ky]
        
    currentMatrix = currentMatrix + currentMatrix.T
        
    for k,val in enumerate(currentMatrix.sum(axis=0)):
        
        if val==0:
            D12[k,k] = 0
        else:    
            D12[k,k] = 1/np.sqrt(2*val)
    
    currentMatrix = np.dot(D12,currentMatrix).dot(D12)
    currentMatrix = (currentMatrix-currentMatrix.min())/(currentMatrix.max()-currentMatrix.min())
    
    return currentMatrix

#Wrapper function for paralellisation 
def Make4DEncoding(Sequence,Block=Blocks,fragments=16):
    
    Sequence = Sequence.seq
    step = len(Sequence)//fragments
    
    container = np.zeros(shape=(fragments,len(Block[1]),len(Block[1]),6))
    toStringSequence = str(Sequence)
    smallSize = len(Block[0])
    
    for k in range(fragments):
        
        localSequence = toStringSequence[k*step:(k+1)*step]
        container[k,:,:,0] = MakeSequenceEncoding(localSequence,Block[1])
        
        stepInner = len(localSequence)//fragments
        localContainer = []
        
        for j in range(fragments):
            
            innerSequence = localSequence[j*stepInner:(j+1)*stepInner]
            localContainer.append(MakeSequenceEncoding(innerSequence,Block[0]))
        
        localContainer = [np.vstack(localContainer[i*smallSize:(i+1)*smallSize]) for i in range(smallSize)]
        localContainer = np.hstack(localContainer)

        container[k,:,:,1] = localContainer
        
        unistep = len(localSequence)//(fragments**2)
        uniContainer = []
        
        for i in range(fragments**2):
            unifrag = localSequence[i*unistep:(i+1)*unistep]
            uniContainer.append(CountUniqueElements(Blocks[0],unifrag))
            
        uniContainer = [np.vstack(uniContainer[i*fragments:(i+1)*fragments]) for i in range(fragments)]    
        uniContainer = np.array(uniContainer)
        container[k,:,:,2::] = uniContainer
        
        
    return container.astype(np.float16)

#Multi purpose parallelisation function 
def GetDataParallel(DataBase,Function): 
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequence Graphs Functions
###############################################################################

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]
sm = 0

for blk in fragmentsDirs:
    
    Container = GetSeqs(blk)
    sm = sm +len(Container)
    print(sm)
    
    if len(Container)>0:
        names = [seq.id for seq in Container]
        data = GetDataParallel(Container,Make4DEncoding)
    
        for nme, db in zip(names,data):
            np.save(matrixData+'/'+nme, db)
