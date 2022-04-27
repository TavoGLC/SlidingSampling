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
import pandas as pd
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

from itertools import product

###############################################################################
# Utility Functions
###############################################################################

#Wrapper function, flatten a list of lists
def ListFlatten(List):
    return [item for sublist in List for item in sublist]

###############################################################################
# Visualization functions
###############################################################################

def PlotStyle(Axes): 
    """
    Applies a general style to a plot 
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
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)
    

###############################################################################
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
            
    localCounter=[val/nSeq for val in localCounter]
    
    return localCounter


def CountUniqueElementsByBlock(Sequences,UniqueElementsBlock,config=False):
    '''
    
    Parameters
    ----------
    Sequences : list, array
        Data set.
    UniqueElementsBlock : list,array
        Unique element collection of different fragment size.
    config : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    Container : array
        Contains the frequeny of each unique element.
    '''
    
    Container=np.array([[],[]])
    
    for k,block in enumerate(UniqueElementsBlock):
        
        countPool=mp.Pool(MaxCPUCount)
        if config:
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val.seq,True )for val in Sequences])
        else:    
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val.seq )for val in Sequences])
        countPool.close()
        
        if k==0:
            Container=np.array(currentCounts)
        else:
            Container=np.hstack((Container,currentCounts))
            
    return Container

###############################################################################
# Global definitions
###############################################################################

GlobalDirectory = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  

seqDataDir=SequencesDir+'/sequences.fasta'
sequencesFrags = SequencesDir + '/splitted'

MaxCPUCount=int(0.85*mp.cpu_count())

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
UniqueElementsBlock = []

maxSize = 6
for k in range(1,maxSize):
    
    UniqueElementsBlock.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
Lenghts=[len(val) for val in UniqueElementsBlock]
Lenghts=np.cumsum(Lenghts)

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]

###############################################################################
# Blocks
###############################################################################

blocksContainer = []

for val in fragmentsDirs:
    
    currentSeqs = GetSeqs(val)
    blocksContainer.append(CountUniqueElementsByBlock(currentSeqs,UniqueElementsBlock))

Kmers = np.vstack(blocksContainer)
Kmers2 = [val for val in blocksContainer if len(val)!=0]
Kmers2 = np.vstack(Kmers2)

###############################################################################
# Blocks
###############################################################################
  
sequenceNames = []

for val in fragmentsDirs:
    
    currentSeqs = GetSeqs(val)
    sequenceNames = sequenceNames + [val.id for val in currentSeqs]
    
KmerDF = pd.DataFrame()
KmerDF['id'] = np.array(sequenceNames)
headers = [val for li in UniqueElementsBlock for val in li]

for k,hd in enumerate(headers):
    KmerDF[hd] = Kmers2[:,k]
    
KmerDF.to_csv(GlobalDirectory+'/KmerDataExt.csv',index=False)
