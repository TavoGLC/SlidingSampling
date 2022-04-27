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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from itertools import product

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
    
###############################################################################
# Custom Layers
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

Blocksb = np.array([item for sublist in Blocks for item in sublist])

###############################################################################
# Paths and constants 
###############################################################################

Model01Path = r'/media/tavoglc/Datasets/datasets/main/models/Model01'

KmerPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
KmerDir = KmerPath+'/KmerDataExt.csv'

MetaDataPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
MetaDataP = MetaDataPath + '/newMetaData.csv'

MetaData = pd.read_csv(MetaDataP)
MetaData = MetaData.set_index('id')

counter = 0
DPI = 300

###############################################################################
# Latent Walk
###############################################################################

modelfile =Model01Path + '/AEFold' +str(1)+'.h5'
AE = load_model(modelfile,custom_objects={'KLDivergenceLayer': KLDivergenceLayer,'Sampling': Sampling})
Decoder = AE.layers[2]

steps = 6
top = 3
vls = np.linspace(-2.0,2.0,num=steps)
fig,axs = plt.subplots(steps,steps,figsize=(25,25),sharex=True,sharey=True)

for k,val in enumerate(vls):
    upval = Decoder(np.array([[0,val]]))[0]
    for j,sal in enumerate(vls):
        if k!=j:
            dta = np.array(upval-Decoder(np.array([[0,sal]]))[0])
            sortIndex = np.argsort(dta)
            armin = sortIndex[0:top]
            armax = sortIndex[::-1][0:top]
            axs[k,j].bar(np.arange(len(dta)),dta,1.2,color='gray',alpha=0.75)
            axs[k,j].bar(armin,dta[armin],2.4,color='red')
            axs[k,j].bar(armax,dta[armax],2.4,color='blue')
            axs[k,j].text(0,-0.4,' - '.join(Blocksb[armin]))
            axs[k,j].text(0,0.4,' - '.join(Blocksb[armax]))
            PlotStyle(axs[k,j])
        else:
            ImageStyle(axs[k,j])

plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Latent Walk
###############################################################################

steps = 100
top = 2
mins = []
maxs = []
vls = np.linspace(-2.0,2.0,num=steps)
upval = Decoder(np.array([[0,-2]]))[0]

for k,val in enumerate(vls):
    dta = np.array(upval-Decoder(np.array([[0,val]]))[0])
    sortIndex = np.argsort(dta)
    armin = sortIndex[0:top]
    armax = sortIndex[::-1][0:top]
    mins = mins + list(Blocksb[armin])
    maxs = maxs + list(Blocksb[armax])

uniquemins = [ val for val in list(set(mins)) if len(val)>3][0:9]
uniquemaxs = [ val for val in list(set(maxs)) if len(val)>3][0:9]

###############################################################################
# Kmer change visalization
###############################################################################

headers = ['id'] + list(Blocksb)

KmerData = pd.read_csv(KmerDir,usecols=headers)
KmerData['id'] = [val[0:-2] for val in KmerData['id']]
KmerData = KmerData.set_index('id')

MetaData = pd.concat([MetaData,KmerData], axis=1)

def MakePanelByBlock(DataFrame,column,block,size=(20,20)):
    
    panelSize = int(np.sqrt(len(block)))
    fig,axs = plt.subplots(panelSize,panelSize,figsize=size,sharex=True,sharey=True)
    axs = axs.ravel()

    for k,val in enumerate(block):
        
        cdata = np.array(DataFrame.groupby(column).mean()[val])
        cdata = (cdata-cdata.min())/(cdata.max()-cdata.min())
        xdata = np.linspace(0,1,num=len(cdata))
        axs[k].plot(xdata,cdata,label=val)
        axs[k].legend()
        axs[k].set_xlabel('Normalized '+column)
        axs[k].set_ylabel('Normalized frequency')
        PlotStyle(axs[k])

def MakePanelBByBlock(DataFrame,column,block,size=(15,10)):
    
    panelSize = int(np.sqrt(len(block)))
    fig,axs = plt.subplots(panelSize,panelSize,figsize=size,sharex=True,sharey=True)
    axs = axs.ravel()

    for k,val in enumerate(block):
        
        xdata = np.array(DataFrame.groupby(column)['lengthofday'].mean())
        xmin,xmax  = xdata.min(),xdata.max()
        xdata = (xdata-xmin)/(xmax-xmin)
        
        ydata = np.array(DataFrame.groupby(column)[val].mean())
        ymin,ymax  = ydata.min(),ydata.max()
        ydata = (ydata-ymin)/(ymax-ymin)
        
        xerr = np.array(DataFrame.groupby(column)['lengthofday'].std())
        xerr[np.isnan(xerr)] = 0
        xerr = xerr/(xmax-xmin)
        
        yerr = np.array(DataFrame.groupby(column)[val].std())
        yerr[np.isnan(yerr)] = 0
        yerr = yerr/(ymax-ymin)
        
        cdata = np.array(DataFrame.groupby(column)[column].mean())
        cdata = (cdata-cdata.min())/(cdata.max()-cdata.min())
        
        colors = [plt.cm.viridis(each) for each in cdata]
        cax = axs[k]
        cax.scatter(xdata,ydata,color=[0,0,0,1],s=1,label=val,alpha=0.5)

        for x,y,xer,yer,col in zip(xdata,ydata,xerr,yerr,colors):
            
            el = cax.add_artist(Ellipse((x,y),xer,yer))
            el.set_clip_box(cax.bbox)
            el.set_alpha(0.75)
            el.set_facecolor(col[0:3])
            
        cax.legend(loc=1)
        
        fig.supxlabel('Normalized Mean Day Duration by '+column)
        fig.supylabel('Normalized Mean Frequency')
        PlotStyle(cax)
        plt.tight_layout()
        
###############################################################################
# Kmer change 
###############################################################################

MakePanelByBlock(MetaData,'dayofyear',Blocks[0])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'outbreaktime',Blocks[0])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'lengthofday',Blocks[0])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'dayofyear',Blocks[1])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'outbreaktime',Blocks[1])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'lengthofday',Blocks[1])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'dayofyear',uniquemins)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'outbreaktime',uniquemins)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'lengthofday',uniquemins)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'dayofyear',uniquemaxs)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'outbreaktime',uniquemaxs)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelByBlock(MetaData,'lengthofday',uniquemaxs)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Kmer change 
###############################################################################

MakePanelBByBlock(MetaData,'dayofyear',Blocks[0])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'outbreaktime',Blocks[0])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1


MakePanelBByBlock(MetaData,'dayofyear',Blocks[1])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'outbreaktime',Blocks[1])
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'dayofyear',uniquemins)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'outbreaktime',uniquemins)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'dayofyear',uniquemaxs)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakePanelBByBlock(MetaData,'outbreaktime',uniquemaxs)
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

