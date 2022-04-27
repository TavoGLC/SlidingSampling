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

###############################################################################
# Loading packages 
###############################################################################

import re
import numpy as np
import pandas as pd
import multiprocessing as mp


from Bio import SeqIO
from io import StringIO

from itertools import product
from collections import Counter
from numpy import linalg as LA

from scipy.spatial import distance as ds

import tensorflow as tf

from keras.models import load_model
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work poperly on those graphics cards 

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
# Custom layers keras
###############################################################################

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.annealing = tf.Variable(10.**-16,dtype=tf.float32,trainable = False)
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*self.annealing*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        self.add_metric(klbatch,name='kl_loss',aggregation='mean')
        
        return inputs

class Sampling(Layer):
    '''
    Custom sampling layer
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}
    
    @tf.autograph.experimental.do_not_convert   
    def call(self,inputs,**kwargs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
BlocksB =  [item for sublist in Blocks for item in sublist]

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


def Make3DBidirectionalEncoding(Sequence,Block=Blocks[1]):
    
    matrix = np.zeros((len(Block),len(Block),2))
    
    localMatrix = MakeBidirectionalEncoding(Sequence,Block,skip=1)
    matrix[:,:,0] = localMatrix[0]
    matrix[:,:,1] = localMatrix[1]
        
    return matrix
        
def GetSeqDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val.seq )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Paths and constants 
###############################################################################

MaxCPUCount=int(0.90*mp.cpu_count())

modelAPath = r'/media/tavoglc/Datasets/datasets/main/models/Model05a'
EncoderAFile = modelAPath + '/EncoderFold0.h5'
AEAFile = modelAPath + '/AEFold0.h5'

modelBPath = r'/media/tavoglc/Datasets/datasets/main/models/Model05'
EncoderBFile = modelBPath + '/EncoderFold0.h5'
AEBFile = modelBPath + '/AEFold0.h5'

GenomePath = r'/media/tavoglc/storage/backup/main2/main/mining/genome/GRCh38_latest_rna.fna'

KmerPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
KmerDir = KmerPath+'/KmerDataExt.csv'

MetaDataPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
MetaDataP = MetaDataPath + '/newMetaData.csv'

MetaData = pd.read_csv(MetaDataP)
MetaData = MetaData.set_index('id')

###############################################################################
# Data
###############################################################################

columns = ['id'] + BlocksB
KmerData = pd.read_csv(KmerDir,usecols = columns)
KmerData['id'] = [val[0:-2] for val in KmerData['id']]
KmerData = KmerData.set_index('id')

threshold = ds.euclidean(KmerData.min(),KmerData.max())
meanSARSCov2 = np.array(KmerData[BlocksB].mean()).reshape(1,-1)

seqs = GetSeqs(GenomePath)
exceptseqs = [76223, 76227, 87731, 111340]

seqs = [val for k,val in enumerate(seqs) if k not in exceptseqs]

###############################################################################
# Filtering by nucleotide composition
###############################################################################

mRNAFreqs = CountUniqueElementsByBlock(seqs,Blocks)

distances = ds.cdist(meanSARSCov2,mRNAFreqs,'euclidean')

selectIndex0 = [k for k,val in enumerate(distances[0]) if val < threshold]
selectedSeqs0 = [seqs[k] for k in selectIndex0]

###############################################################################
# Filtering by temporal component
###############################################################################

AE = load_model(AEAFile,custom_objects={'KLDivergenceLayer': KLDivergenceLayer,'Sampling': Sampling})
Decoder = AE.layers[2]

xs = np.zeros(16)
ys = np.linspace(-6,4,num=16)

walk = np.vstack((xs,ys)).T
walkData = np.array(Decoder(walk))
temporalElements = walkData[:,13:16,:,:,:]

cutoff = 0.1

temporalElements[temporalElements>cutoff] = 1
temporalElements[temporalElements<cutoff] = 0
temporalElements = temporalElements.astype(np.int)

def GetDistances(sample,data=temporalElements):
    
    container = []
    for k in range(data.shape[0]):
        innerContainer = []
        for j in range(data.shape[1]):
            element = data[k,j,:,:,:]
            diff = element-sample
            distance = LA.norm(diff)
            innerContainer.append(distance)
            
        container.append(innerContainer)
        
    return container

encodedData = GetSeqDataParallel(selectedSeqs0,Make3DBidirectionalEncoding)          
distancesData = [GetDistances(val.astype(np.int)) for val in encodedData]
selectIndex1 = [k for k,val in enumerate(distancesData) if np.array(val).min()==0]
selectedSeqs1 = [selectedSeqs0[k] for k in selectIndex1]

###############################################################################
# Filtering by second temporal component
###############################################################################

AEB = load_model(AEBFile,custom_objects={'KLDivergenceLayer': KLDivergenceLayer,'Sampling': Sampling})
DecoderB = AEB.layers[2]

xs = np.zeros(16)
ys = np.linspace(-2,2,num=16)

walkB = np.vstack((ys,xs)).T
walkDataB = np.array(DecoderB(walkB))
temporalElementsB = walkDataB[:,13:16,:,:,:]

encodedData1 = GetSeqDataParallel(selectedSeqs1,Make3DBidirectionalEncoding)
distancesB = [GetDistances(val,data=temporalElementsB) for val in encodedData1]
minDistancesB = [np.min(np.array(val)) for val in distancesB]

disc = np.mean(minDistancesB)-np.std(minDistancesB)
selectIndex2 = [k for k,val in enumerate(minDistancesB) if val<disc]
selectedSeqs2 = [selectedSeqs1[k] for k in selectIndex2]

###############################################################################
# Sequence selection. 
###############################################################################

coding = []
noncoding = []

for val in selectedSeqs2:
    
    if re.match('(.*)ncRNA(.*)',val.description) or re.match('(.*)long non-coding RNA(.*)',val.description) or re.match('(.*)non-coding RNA(.*)',val.description):
        noncoding.append(val)
    else:
        coding.append(val)

notPredicted = []
Predicted = []

for val in coding:
    if re.match('(.*)PREDICTED(.*)',val.description):
        Predicted.append(val)
    else:
        notPredicted.append(val)

ids = [val.id[0:-2] for val in notPredicted]

with open('ids', 'w') as f:
    for val in ids:
        f.write(val)
        f.write('\n')
