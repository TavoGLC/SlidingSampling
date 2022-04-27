
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

import os 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Bio import SeqIO
from io import StringIO

from itertools import product
from collections import Counter
from numpy import linalg as LA

##############################################################################
# Plot functions 
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
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)

def ImageStyle(Axes): 
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
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

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

#Wrapper function for graph visualization 
def MakeSequenceGraph(Sequence,NodeNames,skip=0):

    Nodes=np.arange(len(NodeNames))
    fragmentSize=len(NodeNames[0])
    processedSequence=SplitString(Sequence,fragmentSize)
    
    localGraph=nx.Graph()
    localGraph.add_nodes_from(Nodes)
    x,y = MakeAdjacencyList(processedSequence,NodeNames,skip=skip)
    
    for val,sal in zip(x,y):
        localGraph.add_edge(val,sal)
        
    return localGraph

def RelationalSkip(Sequence,Block,skip=0):
    '''
    Parameters
    ----------
    Sequence : string, biopython seq object
        Sequence to analyze.
    Block : array-like,list
        Contains the unique elements to make the graph.
    skip : int, optional
        Overlap between tokens. The default is 0.

    Returns
    -------
    array
        normalized adjacency matrix.

    '''
    
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


###############################################################################
# Data paths
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  
seqDataDir=SequencesDir+'/sequences.fasta'
sequencesFrags = SequencesDir + '/splitted'
outputpath = r'/media/tavoglc/Datasets/datasets/main/analisis'

counter = 0
DPI = 300

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
###############################################################################
# Sequence examples
###############################################################################

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]
seqs = GetSeqs(fragmentsDirs[0])

testSeq = str(seqs[10].seq)
testSeqReverse = testSeq[::-1]

###############################################################################
# Combinatios within the alphabet
###############################################################################

plt.figure()
for alph in range(2,23,2):
    if alph==4:    
        plt.plot([alph**val for val in range(7)],'b-',label="DNA/RNA")
    if alph==22:    
        plt.plot([alph**val for val in range(7)],'r-',label="Protein")
        
    else:
        plt.plot([alph**val for val in range(7)],color='gray',alpha=0.5)
        
plt.yscale('symlog')
plt.xlabel('Sequence Length')
plt.ylabel('Number of Combinations')
plt.legend()
ax = plt.gca()
PlotStyle(ax)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1


###############################################################################
#Variability inside the sequence 
###############################################################################

nseq = len(testSeq)
containerA = []
containerB = []
containerC = []

for k in range(15):
    localSeq = [testSeq[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/nseq
    containerA.append(ratio)
    
for k in range(0,nseq,100):
    localSeq = [testSeq[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/nseq
    containerB.append(ratio)

for k in range(0,nseq,100):
    localSeq = [testSeq[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/len(localSeq)
    containerC.append(ratio)

fig,axs = plt.subplots(1,3,figsize=(15,7))

_ = axs[0].plot(np.arange(15),containerA)
_ = axs[1].plot(np.arange(0,nseq,100),containerB)
_ = axs[2].plot(np.arange(0,nseq,100),containerC)

axs[0].set_xlabel('Sequence Length')
axs[1].set_xlabel('Sequence Length')
axs[2].set_xlabel('Sequence Length')

axs[0].set_ylabel('Unique Elements (Normalized to Initial Sequence Size)')
axs[1].set_ylabel('Unique Elements (Normalized to Initial Sequence Size)')
axs[2].set_ylabel('Unique Elements (Normalized to Current Sequence Size)')

PlotStyle(axs[0])
PlotStyle(axs[1])
PlotStyle(axs[2])

fig.suptitle('Forward Sequence')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

nseq = len(testSeq)
containerA = []
containerB = []
containerC = []

for k in range(15):
    localSeq = [testSeqReverse[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/nseq
    containerA.append(ratio)
    
for k in range(0,nseq,100):
    localSeq = [testSeqReverse[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/nseq
    containerB.append(ratio)

for k in range(0,nseq,100):
    localSeq = [testSeqReverse[j:j+k+1] for j in range(nseq-k+1)]
    ratio = len(set(localSeq))/len(localSeq)
    containerC.append(ratio)

fig,axs = plt.subplots(1,3,figsize=(15,7))

_ = axs[0].plot(np.arange(15),containerA)
_ = axs[1].plot(np.arange(0,nseq,100),containerB)
_ = axs[2].plot(np.arange(0,nseq,100),containerC)

axs[0].set_xlabel('Sequence Length')
axs[1].set_xlabel('Sequence Length')
axs[2].set_xlabel('Sequence Length')

axs[0].set_ylabel('Unique Elements (Normalized to Initial Sequence Size)')
axs[1].set_ylabel('Unique Elements (Normalized to Initial Sequence Size)')
axs[2].set_ylabel('Unique Elements (Normalized to Current Sequence Size)')

PlotStyle(axs[0])
PlotStyle(axs[1])
PlotStyle(axs[2])

fig.suptitle('Reversed Sequence')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1
###############################################################################
#Relational representation Forward
###############################################################################

alphas = [0.5,0.25,0.125,0.025]
skips = [k for k in range(-1,4)]

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):    
        localGraph = MakeSequenceGraph(testSeq,blk,skip=k)
        nx.draw(localGraph,pos=nx.circular_layout(localGraph),alpha=alphas[j],node_size=20,ax=ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Graph)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):
        
        localAdjacency = RelationalSkip(testSeq,blk,skip=k)
        ax[i,j].imshow(localAdjacency)
        ImageStyle(ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):
        
        localAdjacency = RelationalSkip(testSeq,blk,skip=k)
        ax[i,j].hist(localAdjacency.ravel(),bins=50)
        PlotStyle(ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
#Relational representation Reversded
###############################################################################

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):    
        localGraph = MakeSequenceGraph(testSeqReverse,blk,skip=k)
        nx.draw(localGraph,pos=nx.circular_layout(localGraph),alpha=alphas[j],node_size=20,ax=ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Graph)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):
        localAdjacency = RelationalSkip(testSeqReverse,blk,skip=k)
        ax[i,j].imshow(localAdjacency)
        ImageStyle(ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,ax = plt.subplots(len(skips),len(Blocks),figsize=(20,20))
for i,k in enumerate(skips):    
    for j,blk in enumerate(Blocks):
        localAdjacency = RelationalSkip(testSeqReverse,blk,skip=k)
        ax[i,j].hist(localAdjacency.ravel(),bins=50)
        PlotStyle(ax[i,j])
        ax[i,j].set_title(str(len(blk[0]))+'-mer  skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Forward and reverse patterns  
###############################################################################

###############################################################################
# Forward and reverse patterns  
###############################################################################

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[0],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1
    
fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()    
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[0],skip=k)
    ax.hist(localAdjacency.ravel(),bins=25)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[0],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()    
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[0],skip=k)
    ax.hist(localAdjacency.ravel(),bins=25)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Forward and reverse patterns  
###############################################################################

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[1],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1
    
fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()    
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[1],skip=k)
    ax.hist(localAdjacency.ravel(),bins=25)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[1],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1
    
fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[1],skip=k)
    ax.hist(localAdjacency.ravel(),bins=25)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Forward and reverse patterns  
###############################################################################

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[2],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()    
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeq,Blocks[2],skip=k)
    ax.hist(localAdjacency.ravel(),bins=50)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Forward Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[2],skip=k)
    ax.imshow(localAdjacency)
    ImageStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(4,4,figsize=(20,20))
axs = axs.ravel()    
for k,ax in enumerate(axs):
    localAdjacency = RelationalSkip(testSeqReverse,Blocks[2],skip=k)
    ax.hist(localAdjacency.ravel(),bins=50)
    PlotStyle(ax)
    ax.set_title( 'skip '+str(k+1))
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Reversed Sequence (Adjacency Matrix Distribution)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Sequence Size distribution
###############################################################################
fig,axs = plt.subplots(3,3,figsize=(20,20),sharey=True,sharex=True)
axs = axs.ravel()
fragments = [k*len(Blocks[0])**2 for k in range(3,12)]

for k,val  in enumerate(fragments):
    fragmentForward = testSeq[0:val]
    fragmentReverse = testSeqReverse[0:val]
    forwardAdj = RelationalSkip(fragmentForward,Blocks[0],skip=0)
    reverseAdj = RelationalSkip(fragmentReverse,Blocks[0],skip=0)
    axs[k].hist(forwardAdj.ravel(),bins=20,label='Forward Sequence')
    axs[k].hist(reverseAdj.ravel(),bins=20,label='Reverse Sequence',alpha=0.5)
    axs[k].legend()
    axs[k].set_title( 'Fragmen Size = '+str(val))
    PlotStyle(axs[k])
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Adjacency Matrix Distribution')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(3,3,figsize=(20,20),sharey=True,sharex=True)
axs = axs.ravel()
fragments = [k*len(Blocks[1])**2 for k in range(1,10)]
for k,val  in enumerate(fragments):
    fragmentForward = testSeq[0:val]
    fragmentReverse = testSeqReverse[0:val]
    forwardAdj = RelationalSkip(fragmentForward,Blocks[1],skip=1)
    reverseAdj = RelationalSkip(fragmentReverse,Blocks[1],skip=1)
    axs[k].hist(forwardAdj.ravel(),bins=20,label='Forward Sequence')
    axs[k].hist(reverseAdj.ravel(),bins=20,label='Reverse Sequence',alpha=0.5)
    axs[k].legend()
    axs[k].set_title( 'Fragmen Size = '+str(val))
    PlotStyle(axs[k])
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Adjacency Matrix Distribution')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

fig,axs = plt.subplots(3,3,figsize=(20,20),sharex=True)
axs = axs.ravel()
fragments = [k*len(Blocks[2])**2 for k in range(1,10)]
for k,val  in enumerate(fragments):
    fragmentForward = testSeq[0:val]
    fragmentReverse = testSeqReverse[0:val]
    forwardAdj = RelationalSkip(fragmentForward,Blocks[2],skip=2)
    reverseAdj = RelationalSkip(fragmentReverse,Blocks[2],skip=2)
    axs[k].hist(forwardAdj.ravel(),bins=50,label='Forward Sequence')
    axs[k].hist(reverseAdj.ravel(),bins=50,label='Reverse Sequence',alpha=0.5)
    axs[k].legend()
    axs[k].set_title( 'Fragmen Size = '+str(len(fragmentForward)))
    PlotStyle(axs[k])
fig.subplots_adjust(wspace=0.2, hspace=0.2,top=0.95)
fig.suptitle('Adjacency Matrix Distribution')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Sequence Size statistics
###############################################################################
def DescriptorWrapper(Sequences,block,skip,encoding_function,function,Sizes,Reverse=False):
    nSeqs = len(Sequences)
    
    Container = np.zeros((nSeqs,len(Sizes)))
    
    for k,val in enumerate(Sequences):
        
        if Reverse:    
            localSeq = val[::-1]
        else:
            localSeq = val
        
        for j,bound in enumerate(Sizes):
            
            fragment = localSeq[0:bound]
            localAdjacency = encoding_function(fragment,block,skip=skip)
            Container[k,j] = function(localAdjacency)
            
    return Container,Sizes

def GetDescriptors(Sequences,block,skip,function,Sizes,Reverse=False):
    return DescriptorWrapper(Sequences,block,skip,RelationalSkip,function,Sizes,Reverse=False)

def MakePanel(Data,ax,title):
    for val in Data[0]:
        ax.plot(Data[1],val,color='grey',alpha=0.5)
    ax.plot(Data[1],Data[0].mean(axis=0),color='red')
    ax.set_title(title) 
    ax.set_xlabel('Fragment Size')
    PlotStyle(ax)
    
###############################################################################
# Sequence Size
###############################################################################

index = np.arange(len(seqs))
np.random.shuffle(index)
localSeqs = [str(seqs[k].seq) for k in index[0:25]]

for nme,func in zip(['Mean','Standard Deviation'],[np.mean,np.std]):
    
    fix,axs = plt.subplots(1,2,figsize=(15,7))
    forwardData = GetDescriptors(localSeqs,Blocks[0],0,func,np.arange(100,2500,50))
    reverseData = GetDescriptors(localSeqs,Blocks[0],0,func,np.arange(100,2500,50),Reverse=True)
    MakePanel(forwardData,axs[0],'Forward')
    axs[0].set_ylabel(nme)
    MakePanel(reverseData,axs[1],'Reverse')
    axs[1].set_ylabel(nme)
    fig.suptitle('1-mer 1-skip')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1
        
for nme,func in zip(['Mean','Standard Deviation'],[np.mean,np.std]):
    
    fix,axs = plt.subplots(1,2,figsize=(15,7))
    forwardData = GetDescriptors(localSeqs,Blocks[1],1,func,np.arange(500,5000,100))
    reverseData = GetDescriptors(localSeqs,Blocks[1],1,func,np.arange(500,5000,100),Reverse=True)
    MakePanel(forwardData,axs[0],'Forward')
    axs[0].set_ylabel(nme)
    MakePanel(reverseData,axs[1],'Reverse')
    axs[1].set_ylabel(nme)
    fig.suptitle('2-mer 2-skip')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1
        
for nme,func in zip(['Mean','Standard Deviation'],[np.mean,np.std]):
    
    fix,axs = plt.subplots(1,2,figsize=(15,7))
    forwardData = GetDescriptors(localSeqs,Blocks[2],2,func,np.arange(500,28000,400))
    reverseData = GetDescriptors(localSeqs,Blocks[2],2,func,np.arange(500,28000,400),Reverse=True)
    MakePanel(forwardData,axs[0],'Forward')
    axs[0].set_ylabel(nme)
    MakePanel(reverseData,axs[1],'Reverse')
    axs[1].set_ylabel(nme)
    fig.suptitle('3-mer 3-skip')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1
