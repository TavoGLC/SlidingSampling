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
import tensorflow as tf
import matplotlib.pyplot as plt

from joblib import dump
from itertools import product
from sklearn import preprocessing as pr

from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense, Layer, BatchNormalization

globalSeed=768
from numpy.random import seed 
seed(globalSeed)
tf.compat.v1.set_random_seed(globalSeed)

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
# Network definition
###############################################################################

def MakeDenseCoder(InputShape,Units,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Data shape.
    Units : list
        List with the number of dense units per layer.
    Latent : int
        Size of the latent space.
    UpSampling : bool, optional
        Controls the behaviour of the function, False returns the encoder while True returns the decoder. 
        The default is False.

    Returns
    -------
    InputFunction : Keras Model input function
        Input Used to create the coder.
    localCoder : Keras Model Object
        Keras model.

    '''
    Units.append(Latent)
    
    if UpSampling:
        denseUnits=Units[::-1]
        Name="Decoder"
    else:
        denseUnits=Units
        Name="Encoder"
    
    InputFunction=Input(shape=InputShape)
    nUnits=len(denseUnits)
    X=Dense(denseUnits[0],use_bias=False)(InputFunction)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    
    for k in range(1,nUnits-1):
        X=Dense(denseUnits[k],use_bias=False)(X)
        X=BatchNormalization()(X)
        X=Activation('relu')(X)
    
    X=Dense(denseUnits[-1],use_bias=False)(X)
    X=BatchNormalization()(X)
    
    if UpSampling:
        Output=Activation('sigmoid')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
    else:    
        Output=Activation('relu')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
    
    return InputFunction,localCoder

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.annealing = tf.Variable(10**-16,dtype=tf.float32,trainable = False)
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

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder 
def MakeVariationalNetwork(Latent):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer()([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck

#Wrapper function, creates the dense autoencoder
def MakeDenseAutoencoder(InputShape,Units,Latent):
    
    InputEncoder,Encoder=MakeDenseCoder(InputShape,Units,Latent)
    InputDecoder,Decoder=MakeDenseCoder((Latent,),Units,Latent,UpSampling=True)
    AEoutput=Decoder(Encoder(InputEncoder))
    AE=Model(inputs=InputEncoder,outputs=AEoutput)
    
    return Encoder,Decoder,AE

#Wrapper function, merges the dense autoencoder and the variational 
#layers 
def MakeVariationalDenseAutoencoder(InputShape,Units,Latent):
    
    InputEncoder,Encoder=MakeDenseCoder(InputShape,Units,Latent)
    InputVAE,VAE=MakeVariationalNetwork(Latent)
    InputDecoder,Decoder=MakeDenseCoder((Latent,),Units,Latent,UpSampling=True)
    
    VAEencoderOutput=VAE(Encoder(InputEncoder))
    VAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=Decoder(VAEencoder(InputEncoder))
    VAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return VAEencoder,Decoder,VAEAE

###############################################################################
# Auxiliary functions
###############################################################################

#Custom callback, scales the KL-loss 
class KLAnnealing(keras.callbacks.Callback):

    def __init__(self,position, weigths):
        super().__init__()
        self.position = position
        self.weigths = tf.Variable(weigths,trainable=False,dtype=tf.float32)

    def on_epoch_end(self, epoch,logs=None):
        
        weights = self.model.get_weights()
        weights[self.position] = self.weigths[epoch]
        self.model.set_weights(weights)

###############################################################################
# Data loading 
###############################################################################

foldsPath = r'/media/tavoglc/Datasets/datasets/main/splits'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv(foldsPath+'/validation.csv')

Complete = pd.read_csv(foldsPath+'/complete.csv')

DataPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
DataDir = DataPath+'/KmerDataExt.csv'

outputPath = r'/media/tavoglc/Datasets/datasets/main/models/Model01'

###############################################################################
# Data selection
###############################################################################

Alphabet = ['A','C','T','G']
KmerLabels = []

maxSize = 5
for k in range(1,maxSize):
    
    KmerLabels.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
KmerLabels = [item for sublist in KmerLabels for item in sublist]
headers = ['id'] + KmerLabels 

KmerData = pd.read_csv(DataDir,usecols=headers)
KmerData['id'] = [val[0:-2] for val in KmerData['id']]
KmerData = KmerData.set_index('id')

###############################################################################
# Network hyperparameters
###############################################################################

Units = [340,240,140,40,20,10,5,3]
Latent = 2

sh = 0.0001
lr = 0.0025
minlr = 0.00001
batchSize = 256
epochs = 70
decay = 2*(lr-minlr)/epochs
InputShape = 340
AnnealingWeights = sh*np.ones(epochs+1)

###############################################################################
# Kfold cross validation
###############################################################################

ScalersContainer = []
EncoderContainer = []
AEContainer = []
HistoryContainer = []

foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']

for fold in foldNames:
    
    trainLabels = TrainFolds[fold]
    testLabels = TestFolds[fold]
    
    trainData = np.array(KmerData.loc[trainLabels])
    testData = np.array(KmerData.loc[testLabels])
    
    scaler = pr.MinMaxScaler()
    scaler.fit(trainData)
    
    trainData = scaler.transform(trainData)
    testData = scaler.transform(testData)
    
    datasetTrain = tf.data.Dataset.from_tensor_slices((trainData,trainData))
    datasetTest = tf.data.Dataset.from_tensor_slices((testData,testData))
    
    datasetTrain = datasetTrain.shuffle(buffer_size=100,seed=125)
    datasetTrain = datasetTrain.batch(batchSize)
    datasetTrain = datasetTrain.prefetch(2)
    
    datasetTest = datasetTest.shuffle(buffer_size=100,seed=125)
    datasetTest = datasetTest.batch(batchSize)
    datasetTest = datasetTest.prefetch(2)
    
    Encoder,Decoder,AE = MakeVariationalDenseAutoencoder(InputShape,Units,Latent)
    KLAposition = [k for k,val in enumerate(AE.get_weights()) if len(val.shape)==0][0]

    AE.compile(Adam(learning_rate=lr,decay=decay),loss='mse')
    history = AE.fit(datasetTrain,epochs=epochs,
                        validation_data=datasetTest,callbacks=[KLAnnealing(KLAposition,AnnealingWeights)])
    
    ScalersContainer.append(scaler)
    EncoderContainer.append(Encoder)
    AEContainer.append(AE)
    HistoryContainer.append(history)
    
    dump(scaler,outputPath + '/scaler'+fold+'.joblib')
    AE.save(outputPath + '/AE'+fold+'.h5')
    Encoder.save(outputPath + '/Encoder'+fold+'.h5')

    tf.compat.v1.set_random_seed(globalSeed)

###############################################################################
# Learning curves
###############################################################################

RepresentationContainer = []

fig,axs = plt.subplots(2,5,figsize=(30,15))

for k,hist in enumerate(HistoryContainer):
    
    axs[0,k].plot(hist.history['loss'],'k-',label = 'Loss')
    axs[0,k].plot(hist.history['val_loss'],'r-',label = 'Validation Loss')
    axs[0,k].title.set_text('Reconstruction loss')
    PlotStyle(axs[0,k])
    
    axs[1,k].plot(hist.history['kl_loss'],'k-',label = 'Loss')
    axs[1,k].plot(hist.history['val_kl_loss'],'r-',label = 'Validation Loss')
    axs[1,k].title.set_text('Kullback–Leibler loss')
    PlotStyle(axs[1,k])
    
plt.tight_layout()
plt.savefig(outputPath+'/figtraining.png')

###############################################################################
# Latent space visualization
###############################################################################

valDataFrame = pd.DataFrame()
valDataFrame['ids'] = Validation['validation']

validationData = np.array(KmerData.loc[Validation['validation']])
performance = []

fig,axs = plt.subplots(1,5,figsize=(30,15))

for k,block in enumerate(zip(ScalersContainer,EncoderContainer,AEContainer)):
    
    sclr,enc,ae = block
    
    valData = sclr.transform(validationData)
    datasetVal = tf.data.Dataset.from_tensor_slices((valData,valData))
    
    datasetVal = datasetVal.batch(batchSize)
    datasetVal = datasetVal.prefetch(2)
    
    performance.append(ae.evaluate(datasetVal))
    VariationalRepresentation = enc.predict(datasetVal)
    valDataFrame['Dim0_model'+str(k)] = VariationalRepresentation[:,0]
    valDataFrame['Dim1_model'+str(k)] = VariationalRepresentation[:,1]
    
    axs[k].scatter(VariationalRepresentation[:,0],VariationalRepresentation[:,1],alpha=0.15)
    axs[k].title.set_text('Latent Space (model = ' + str(k) +')')
    PlotStyle(axs[k])
    
plt.tight_layout()  
plt.savefig(outputPath+'/figls.png')

valDataFrame.to_csv(outputPath+'/ValDimReduction.csv')

###############################################################################
# Out of sample performance 
###############################################################################

plt.figure()
plt.bar(np.arange(len(performance)),[val[0] for val in performance])
ax = plt.gca()
ax.set_ylabel('Reconstruction Loss',size=13)
ax.set_xlabel('Folds',size=13)
PlotStyle(ax)
plt.savefig(outputPath+'/recloss.png')

plt.figure()
plt.bar(np.arange(len(performance)),[val[1] for val in performance])
ax = plt.gca()
ax.set_ylabel('KL Loss',size=13)
ax.set_xlabel('Folds',size=13)
PlotStyle(ax)
plt.savefig(outputPath+'/KLloss.png')

###############################################################################
# Complete metadata dataset 
###############################################################################

completeDataFrame = pd.DataFrame()
completeDataFrame['ids'] = Complete['complete']

completeData = np.array(KmerData.loc[Complete['complete']])

for k,block in enumerate(zip(ScalersContainer,EncoderContainer,AEContainer)):
    
    sclr,enc,ae = block
    
    comData = sclr.transform(completeData)
    datasetCom = tf.data.Dataset.from_tensor_slices((comData,comData))
    
    datasetCom = datasetCom.batch(batchSize)
    datasetCom = datasetCom.prefetch(2)
    
    VariationalRepresentation = enc.predict(datasetCom)
    
    completeDataFrame['Dim0_model'+str(k)] = VariationalRepresentation[:,0]
    completeDataFrame['Dim1_model'+str(k)] = VariationalRepresentation[:,1]
    
completeDataFrame.to_csv(outputPath+'/CompDimReduction.csv')
