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
import matplotlib.gridspec as gridspec

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
# Network definition
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
# Data paths
###############################################################################

Model01Path = r'/media/tavoglc/Datasets/datasets/main/models/Model01'
Model02Path = r'/media/tavoglc/Datasets/datasets/main/models/Model02'
Model03Path = r'/media/tavoglc/Datasets/datasets/main/models/Model03'
Model04Path = r'/media/tavoglc/Datasets/datasets/main/models/Model04'
Model05Path = r'/media/tavoglc/Datasets/datasets/main/models/Model05'
Model05aPath = r'/media/tavoglc/Datasets/datasets/main/models/Model05a'

MetaDataPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"

Model01Validation = Model01Path + '/ValDimReduction.csv'
Model01Complete = Model01Path + '/CompDimReduction.csv'

Model02Validation = Model02Path + '/ValDimReduction.csv'
Model02Complete = Model02Path + '/CompDimReduction.csv'

Model03Validation = Model03Path + '/ValDimReduction.csv'
Model03Complete = Model03Path + '/CompDimReduction.csv'

Model04Validation = Model04Path + '/ValDimReduction.csv'
Model04Complete = Model04Path + '/CompDimReduction.csv'

Model05Validation = Model05Path + '/ValDimReduction.csv'
Model05Complete = Model05Path + '/CompDimReduction.csv'

Model05aValidation = Model05aPath + '/ValDimReduction.csv'
Model05aComplete = Model05aPath + '/CompDimReduction.csv'

MetaDataP = MetaDataPath + '/newMetaData.csv'
TemperatureP = MetaDataPath + '/temperature2021.csv'
PressureP = MetaDataPath + '/pressure2021.csv'
LocationsP = MetaDataPath + '/locations.csv'

KmerPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
KmerDir = KmerPath+'/KmerDataExt.csv'

###############################################################################
# Data loading 
###############################################################################

Model01ValData = pd.read_csv(Model01Validation)
Model01ValData = Model01ValData.set_index('ids')

Model01CompData = pd.read_csv(Model01Complete)
Model01CompData = Model01CompData.set_index('ids')

Model02ValData = pd.read_csv(Model02Validation)
Model02ValData = Model02ValData.set_index('ids')

Model02CompData = pd.read_csv(Model02Complete)
Model02CompData = Model02CompData.set_index('ids')

Model03ValData = pd.read_csv(Model03Validation)
Model03ValData = Model03ValData.set_index('ids')

Model03CompData = pd.read_csv(Model03Complete)
Model03CompData = Model03CompData.set_index('ids')

Model04ValData = pd.read_csv(Model04Validation)
Model04ValData = Model04ValData.set_index('ids')

Model04CompData = pd.read_csv(Model04Complete)
Model04CompData = Model04CompData.set_index('ids')

Model05ValData = pd.read_csv(Model05Validation)
Model05ValData = Model05ValData.set_index('ids')

Model05CompData = pd.read_csv(Model05Complete)
Model05CompData = Model05CompData.set_index('ids')

Model05aValData = pd.read_csv(Model05aValidation)
Model05aValData = Model05aValData.set_index('ids')

Model05aCompData = pd.read_csv(Model05aComplete)
Model05aCompData = Model05aCompData.set_index('ids')

MetaData = pd.read_csv(MetaDataP)
MetaData = MetaData.set_index('id')

headers = ['id'] + ['A','C','T','G']
KmerData = pd.read_csv(KmerDir,usecols=headers)
KmerData['id'] = [val[0:-2] for val in KmerData['id']]

KmerData = KmerData.set_index('id')

counter = 0
DPI = 300

###############################################################################
# Nucleotide frequency
###############################################################################

MetaData['A'] = [KmerData.loc[val]['A'] for val in MetaData.index]
MetaData['C'] = [KmerData.loc[val]['C'] for val in MetaData.index]
MetaData['T'] = [KmerData.loc[val]['T'] for val in MetaData.index]
MetaData['G'] = [KmerData.loc[val]['G'] for val in MetaData.index]

###############################################################################
# Meta Data Enbcoding 
###############################################################################

def GetSimplifiedStrain(pangoLineage):
    
    if pangoLineage[0]=='A' or pangoLineage[0]=='B':
        return pangoLineage
    else:
        return 'Non'

def GetBinaryStrain(pangoLineage):
    
    if pangoLineage[0]=='A' or pangoLineage[0]=='B':
        return pangoLineage[0]
    else:
        return 'Non'

uniquePango = [str(val) for val in set(MetaData['Pangolin'])]
uniqueSimplified = set([GetSimplifiedStrain(val) for val in uniquePango])

Npango = len(uniquePango)
NsimplifiedPango = len(uniqueSimplified)

pangoToval = dict([(val,sal) for val,sal in zip(uniquePango,np.linspace(0,1,num=Npango)) ])
simpToVal = dict([(val,sal) for val,sal in zip(uniqueSimplified,np.linspace(0,1,num=NsimplifiedPango)) ])

MetaData['pango_encoding'] = [pangoToval[str(val)] for val in MetaData['Pangolin']]
MetaData['simpPango_encoding'] = [simpToVal[GetSimplifiedStrain(str(val))] for val in MetaData['Pangolin']]

binaryToval = dict([('A',0),('B',0.5),('Non',1)])

MetaData['pango_binary'] = [GetBinaryStrain(str(val)) for val in MetaData['Pangolin']]
MetaData['binary_encoding'] = [binaryToval[str(val)] for val in MetaData['pango_binary']]

###############################################################################
# Weather Data
###############################################################################

MetaData['week'].fillna(1,inplace=True)

temperature = pd.read_csv(TemperatureP)
pressure = pd.read_csv(PressureP)
locations = pd.read_csv(LocationsP)

temperature = temperature.set_index('week')
pressure = pressure.set_index('week')

locations = locations[['geo_lat','geo_long']]
uniqueLocations = np.array([list(x) for x in set(tuple(x) for x in np.array(locations))])
locationToIndex = dict([(tuple(sal),val) for val,sal in zip(np.arange(uniqueLocations.shape[0]),uniqueLocations)])

MetaData['week2'] = [int(val*53) for val in MetaData['week']]
MetaData[MetaData['week2'] == 0 ]=1

def GetWeather(index):
    
    localFrame = MetaData.loc[index][['geo_lat','geo_long','week2']]
    temperatureTest = []
    pressureTest = []
    
    for k,val in enumerate(np.array(localFrame)):
        location = tuple([val[0],val[1]])
        if location in locationToIndex.keys():
            index = locationToIndex[location]
            week = val[2]
            temperatureTest.append(temperature[str(index)].loc[week])
            pressureTest.append(pressure[str(index)].loc[week])
        else:
            temperatureTest.append(temperature.loc[week].mean())
            pressureTest.append(pressure.loc[week].mean())
            
    DryAirDensity = [val/(287.05*(sal+273)) for val,sal in zip(pressureTest,temperatureTest)]
    
    return temperatureTest, pressureTest,DryAirDensity

###############################################################################
# Data visualization 
###############################################################################

def MakeThemePlots(Cols,Names,Dataset,Labels,Title='Test',size=(20,10)):
        
    fig,axs = plt.subplots(1,len(Names),figsize=size)
        
    for k,nme in enumerate(Cols):
        
        localColor = MetaData.loc[Dataset.index][nme]
        localMean= localColor.mean()
        locaslStd = localColor.std()
        zscore = [(val-localMean)/locaslStd for val in localColor]
        noOutliersColor = [localMean if np.abs(sal)>4 else val for val,sal in zip(localColor,zscore)]

        axs[k].scatter(Dataset[Labels[0]],Dataset[Labels[1]],c=noOutliersColor,alpha=0.15)
        axs[k].title.set_text(Names[k])
        PlotStyle(axs[k])
    fig.suptitle(Title)

def MakeWeatherThemePlots(Datasets,Labels,Title='Test',size=(20,15)):
    
    Names = ['Temperature','Pressure','Dry Air Density']
        
    weatherDta = GetWeather(Datasets.index)
    fig,axs = plt.subplots(3,len(Names),figsize=size)
        
    for k,nme in enumerate(weatherDta):
        
        localColor = nme
        localMean= np.mean(localColor)
        locaslStd = np.std(localColor)
        zscore = [(val-localMean)/locaslStd for val in localColor]
        noOutliersColor = [localMean if np.abs(sal)>5 else val for val,sal in zip(localColor,zscore)]
            
        axs[0,k].scatter(Datasets[Labels[0]],Datasets[Labels[1]],c=noOutliersColor,alpha=0.15)
        axs[0,k].title.set_text(Names[k])
        PlotStyle(axs[0,k])
        axs[1,k].scatter(MetaData.loc[Datasets.index]['outbreaktime'],noOutliersColor,c=Datasets[Labels[0]],alpha=0.15)
        axs[1,k].title.set_text(Names[k])
        PlotStyle(axs[1,k])
        axs[2,k].scatter(MetaData.loc[Datasets.index]['outbreaktime'],noOutliersColor,c=Datasets[Labels[1]],alpha=0.15)
        axs[2,k].title.set_text(Names[k])
        PlotStyle(axs[2,k])
    fig.suptitle(Title)

###############################################################################
# Time data 
###############################################################################

Datasets = [Model01ValData,Model02ValData,Model03ValData,Model04ValData,Model05ValData,Model05aValData]

fold = 4

Labels = [['Dim0_model'+str(fold),'Dim1_model'+str(fold)],['Dim0_model'+str(fold),'Dim1_model'+str(fold)],['Dim0_model'+str(fold),'Dim1_model'+str(fold)],['Dim0_model'+str(fold),'Dim1_model'+str(fold)],['Dim0_model'+str(fold),'Dim1_model'+str(fold)],['Dim0_model'+str(fold),'Dim1_model'+str(fold)]]

timeCols = ['outbreaktime','dayofyear','week']
Names = ['Outbreak Time', 'Day Of Year', 'Week']

for k,block in enumerate(zip(Datasets,Labels)):
    
    dst,labs = block
    MakeThemePlots(timeCols,Names,dst,labs,Title='Temporal Variables (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1


###############################################################################
# Location data 
###############################################################################

envCols = ['geo_lat','geo_long','geo_alt']
Names = ['Latitude','Longitude','Altitude']

for k,block in enumerate(zip(Datasets,Labels)):
    
    dst,labs = block
    MakeThemePlots(envCols,Names,dst,labs,Title='Location Variables (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1

###############################################################################
# Nucleotide Data 
###############################################################################

envCols = ['A','G','C','T']
Names = ['Adenine','Guanine','Cytosine','Thymine/Uracil']

for k,block in enumerate(zip(Datasets,Labels)):
    
    dst,labs = block
    MakeThemePlots(envCols,Names,dst,labs,Title='Nucleotide Frequency (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1

###############################################################################
# WeatherData
###############################################################################

for k,block in enumerate(zip(Datasets,Labels)):
    
    dst,labs = block
    MakeWeatherThemePlots(dst,labs,Title='Weather Variables (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1
    
###############################################################################
# Lineage
###############################################################################

envCols = ['simpPango_encoding','binary_encoding']
Names = ['Pangolin Lineage', 'Main Pangoline Branch']

for k,block in enumerate(zip(Datasets,Labels)):
    
    dst,labs = block
    MakeThemePlots(envCols,Names,dst,labs,Title='Variants (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1

###############################################################################
# Sample Data 
###############################################################################

compCols = ['pcr_ct','host_age','host_sex_v','host_race_v']
Names = ['PCR Cycles Threshold', 'Host Age','Host Sex','Host Ethnicity']

Datasetsc = [Model01CompData,Model02CompData,Model03CompData,Model04CompData,Model05CompData,Model05aCompData]

for k,block in enumerate(zip(Datasetsc,Labels)):
    
    dst,labs = block
    MakeThemePlots(compCols,Names,dst,labs,Title='Sample Variables (Model 0' + str(k+1) +')')
    plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
    plt.close()
    counter = counter+1

###############################################################################
# Latent Space Walk 
###############################################################################

def MakeGenerativePanel(Decoder,Data,rev=False,location=0,figname='test',size=(14,14)):
    
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(nrows=18, ncols=22)
    # SARS Cov 2 Genome 
    #Michel, C.J., Mayer, C., Poch, O. et al. Characterization of accessory 
    #genes in coronavirus genomes. Virol J 17, 131 (2020). 
    #https://doi.org/10.1186/s12985-020-01402-1
    
    x,y,z = Data
    
    labels = ['ORF1a','ORF1b','S','ORF3a','ORF3bc','E','M','ORF6','ORF7a','ORF7bc','ORF8','ORFN','ORF9bc ','ORF9cc','ORF10']
    sizes = [13203,8086,3849,828,172,228,669,186,366,130,366,1260,294,222,117]
    csizes = np.cumsum(sizes)
    colors = [plt.cm.viridis(val) for val in np.linspace(0,1,num=len(sizes))]
    np.random.shuffle(colors)
    ax = fig.add_subplot(gs[0:2,0:-6])
    ax.barh('GQ', sizes[0], 0.5, label=labels[0],ec=colors[0],fc=colors[0])
    for k in range(1,len(sizes)):
        ax.barh('GQ', sizes[k], 0.5, label=labels[k],left=csizes[k-1],ec=colors[k],fc=colors[k])
    ax.set_xlim([0,30000])
    ax.legend(bbox_to_anchor=(1.0, 1.0),ncol=3)
    ImageStyle(ax)
    
    #Generative Walk
    yvals = np.linspace(np.max(y),np.min(y), num=16)
    for j,val in enumerate(yvals):
        if rev:    
            locIms = Decoder(np.array([[val,0]]))
        else:
            locIms = Decoder(np.array([[0,val]]))
        for k in range(16):
            ax = fig.add_subplot(gs[j+2,k])
            ax.imshow(locIms[0,k,:,:,location])
            ImageStyle(ax)
            
    #Latent Space
    ax = fig.add_subplot(gs[2:,-6::])
    im = ax.scatter(x,y,c=z,alpha=0.15)
    cbar = plt.colorbar(im,ax=ax,fraction=0.05,aspect=60)
    cbar.solids.set(alpha=1)
    ImageStyle(ax)
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)
    fig.suptitle(figname)

###############################################################################
# Latent Space Walk 
###############################################################################

modelfilec =Model05Path + '/AEFold' +str(fold)+'.h5'
AEc = load_model(modelfilec,custom_objects={'KLDivergenceLayer': KLDivergenceLayer,'Sampling': Sampling})
Decoder0 = AEc.layers[2]

PanelData = [Model05ValData['Dim1_model'+str(fold)],Model05ValData['Dim0_model'+str(fold)],MetaData.loc[Model05ValData.index]['week']]

MakeGenerativePanel(Decoder0,PanelData,rev=True,figname='Latent Space Walk (Forward Encoding)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakeGenerativePanel(Decoder0,PanelData,rev=True,location=1,figname='Latent Space Walk (Reverse Encoding)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

###############################################################################
# Latent Space Walk 
###############################################################################

modelfile =Model05aPath + '/AEFold' +str(fold)+'.h5'
AE = load_model(modelfile,custom_objects={'KLDivergenceLayer': KLDivergenceLayer,'Sampling': Sampling})
Decoder = AE.layers[2]

PanelData = [Model05aValData['Dim0_model'+str(fold)],Model05aValData['Dim1_model'+str(fold)],MetaData.loc[Model05aValData.index]['week']]

MakeGenerativePanel(Decoder,PanelData,figname='Latent Space Walk (Forward Encoding)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1

MakeGenerativePanel(Decoder,PanelData,location=1,figname='Latent Space Walk (Reverse Encoding)')
plt.savefig('figure' + str(counter) +'.png', bbox_inches='tight',dpi=DPI)
plt.close()
counter = counter+1
