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

import os
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense,Add,concatenate,LeakyReLU
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Layer,GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization,GlobalMaxPooling2D

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
# Network definition
###############################################################################

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

class SpatialAttention(Layer):
    '''
    Custom Spatial attention layer
    
    '''
    
    def __init__(self,size, **kwargs):
        super(SpatialAttention, self).__init__()
        self.size = size
        self.kwargs = kwargs
        
    def get_config(self):
        cfg = super().get_config()
        return cfg  

    def build(self, input_shapes):
        self.conv = Conv2D(filters=1, kernel_size=self.size, strides=1, padding='same')

    def call(self, inputs):
        pooled_channels = tf.concat(
            [tf.math.reduce_max(inputs, axis=3, keepdims=True),
            tf.math.reduce_mean(inputs, axis=3, keepdims=True)],
            axis=3)

        scale = self.conv(pooled_channels)
        scale = tf.math.sigmoid(scale)

        return inputs * scale

class ChannelAttention(Layer):
    '''
    Custom Channel attention layer
    '''
    
    def __init__(self,**kwargs):
        super(ChannelAttention,self).__init__()
        self.kwargs = kwargs
    
    def get_config(self):
        cfg = super().get_config()
        return cfg
    
    def build(self,input_shape):
        channel = input_shape[-1]
        self.dense0 = Dense(channel)
        self.dense1 = Dense(channel)
    
    def call(self,inputs):
        
        channel = inputs.get_shape().as_list()[-1]
        
        avgpool = GlobalAveragePooling2D()(inputs)
        avgpool = Reshape((1,1,channel))(avgpool)
        avgpool = self.dense0(avgpool)
        avgpool = self.dense1(avgpool)
        
        maxpool = GlobalMaxPooling2D()(inputs)
        maxpool = Reshape((1,1,channel))(maxpool)
        maxpool = self.dense0(maxpool)
        maxpool = self.dense1(maxpool)
        
        feature = Add()([avgpool,maxpool])
        feature = Activation('sigmoid')(feature)
        
        return inputs*feature
    
###############################################################################
# Network definition 
###############################################################################

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder 
def MakeVariationalNetwork(Latent):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer(name='KLDivergence')([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck

def MakeBottleneck(InputShape,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        input shape of the previous convolutional layer.
    Latent : int
        Dimentionality of the latent space.
    UpSampling : bool, optional
        Controls the sampling behaviour of the network.
        The default is False.

    Returns
    -------
    InputFunction : Keras functional model input
        input of the network.
    localCoder : Keras functional model
        Coder model, transition layer of the bottleneck.

    '''
    
    productUnits = np.product(InputShape)
    Units = [productUnits,productUnits//4,productUnits//16,Latent]
    
    if UpSampling:
        finalUnits = Units[::-1]
        InputFunction = Input(shape=(Latent,))
        X = Dense(finalUnits[0],use_bias=False)(InputFunction)
    
    else:
        finalUnits = Units
        InputFunction = Input(shape=InputShape)
        X = Flatten()(InputFunction)
        X = Dense(finalUnits[0],use_bias=False)(X)
                
    
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    for k in range(1,len(Units)-1):
        
        X = Dense(finalUnits[k],use_bias=False)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
    
    X = Dense(finalUnits[-1],use_bias=False)(X)
    
    if UpSampling:
        X=LeakyReLU()(X)
        Output=Reshape(InputShape)(X)
    else:
        Output=LeakyReLU()(X)
        
    Bottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,Bottleneck

#Wrapper function to make the basic convolutional block 
def MakeConvolutionBlock(X, Convolutions):
    
    X = Conv2D(Convolutions, (3, 3), padding='same',use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)

    return X

#Wrapper function to make the dense convolutional block
def MakeDenseBlock(x, Convolutions,Depth):

    concat_feat= x
    for i in range(Depth):
        x = MakeConvolutionBlock(concat_feat,Convolutions)
        concat_feat=concatenate([concat_feat,x])

    return concat_feat

#Wraper function creates a dense convolutional block and resamples the data 
def SamplingBlock(X,Units,Depth,UpSampling=False):
    
    X = MakeDenseBlock(X,Units,Depth)
    
    if UpSampling:
        X = Conv2DTranspose(Units,(3,3),strides=(2,2),padding='same',use_bias=False)(X)
    else:    
        X = Conv2D(Units,(3,3),strides=(2,2),padding='same',use_bias=False)(X)
    
    X = ChannelAttention()(X)
    X = SpatialAttention(3)(X)
    
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    return X 

#Creates the main body of the convolutional auntoencoder
def CoderByBlock(InputShape,Units,Depth,UpSampling=False):
    
    if UpSampling:
        Units=Units[::-1]
    else:
        Units=Units
    
    InputFunction = Input(shape=InputShape)
    X = SamplingBlock(InputFunction,Units[0],Depth,UpSampling=UpSampling)
    
    for k in range(1,len(Units)-1):
        
        if Depth-k+1 <= 1:
            blockSize = 2
        else:
            blockSize = Depth-k
        
        X = SamplingBlock(X,Units[k],blockSize,UpSampling=UpSampling)
        
    if UpSampling:
        X = Conv2D(1,(3,3),padding='same',use_bias=False)(X)
        Output = Activation('sigmoid')(X)
    else:
        X = Conv2D(Units[-1],(3,3),padding='same',use_bias=False)(X)
        Output = LeakyReLU()(X)
        
    coderModel = Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,coderModel

#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs):
    
    InputEncoder,Encoder=CoderFunction(InputShape,Units,BlockSize,**kwargs)
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=CoderFunction(BottleneckInputShape,Units,BlockSize,UpSampling=True,**kwargs)
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeVariationalAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs)
    
    InputVAE,VAE=MakeVariationalNetwork(2)
    VAEencoderOutput=VAE(ConvEncoder(InputEncoder))
    ConvVAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=ConvDecoder(ConvVAEencoder(InputEncoder))
    ConvVAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return InputEncoder,InputDecoder,ConvVAEencoder,ConvDecoder,ConvVAEAE    

###############################################################################
# Auxiliary functions
###############################################################################

#Data sequence to load the data from files 
class DataSequence(Sequence):
    
    def __init__(self, x_set,batch_size):
        self.x = x_set
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x)/self.batch_size))
    
    def __data_generation(self, dirList):
         
        X = np.array([np.load(val) for val in dirList])
        X = X.reshape((-1,64,80,1))
        y = X

        return X,y
    
    def __getitem__(self,idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(batch_x)
        
        return X,y

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


def MakeAnnealingWeights(epochs,cycles,scale=1):
    '''
    Parameters
    ----------
    epochs : int
        min size of the array to return.
    cycles : int
        number of annealing cycles.
    scale : float, optional
        scales the annealing weights. The default is 1.

    Returns
    -------
    array
        annealing weights.

    '''
    pointspercycle = epochs//cycles
    AnnealingWeights = 1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle))))
    
    for k in range(cycles-1):
        AnnealingWeights = np.append(AnnealingWeights,1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle+1)))))
        
    return scale*AnnealingWeights

###############################################################################
# Data loading 
###############################################################################

foldsPath = r'/media/tavoglc/Datasets/datasets/main/splits'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv(foldsPath+'/validation.csv')

Complete = pd.read_csv(foldsPath+'/complete.csv')

outputPath = r'/media/tavoglc/Datasets/datasets/main/models/Model03'

GlobalDirectory=r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
matrixData = GlobalDirectory + '/featuresdata'

fileNames = os.listdir(matrixData)

###############################################################################
# Network hyperparameters
###############################################################################

input_shape = (64,80,1)
Arch = [3,6,9,12,15]
Depth = 4

lr = 0.0001
minlr = 0.000001
epochs = 70
batch_size = 256
decay = 2*(lr-minlr)/epochs

AnnealingWeights = MakeAnnealingWeights(epochs,4,scale=0.00001)

###############################################################################
# Kfold cross validation
###############################################################################

EncoderContainer = []
AEContainer = []
HistoryContainer = []

foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']

for fold in foldNames:
    
    trainNames = [val+'.1.npy' for val in TrainFolds[fold]]
    trainNames = list(set(trainNames).intersection(fileNames))
    trainPaths = np.array([matrixData+'/'+val for val in trainNames]) 
    
    testNames = [val+'.1.npy' for val in TestFolds[fold]]
    testNames = list(set(testNames).intersection(fileNames))
    testPaths = np.array([matrixData+'/'+val for val in testNames])   

    datasetTrain = tf.data.Dataset.from_generator(DataSequence, 
                                                     args=[trainPaths, batch_size],
                                                     output_types=(tf.float32, tf.float32 ), 
                                                     output_shapes=((None,64,80,1), (None,64,80,1)))

    datasetTrain = datasetTrain.shuffle(buffer_size=100,seed=125)
    datasetTrain = datasetTrain.prefetch(2)
    
    datasetTest = tf.data.Dataset.from_generator(DataSequence, 
                                                     args=[testPaths, batch_size],
                                                     output_types=(tf.float32, tf.float32 ), 
                                                     output_shapes=((None,64,80,1), (None,64,80,1)))
    
    datasetTest = datasetTest.shuffle(buffer_size=100,seed=125)
    datasetTest = datasetTest.prefetch(2)
    
    _,_,Encoder,Decoder,AE = MakeVariationalAutoencoder(CoderByBlock,input_shape,Arch,Depth)
    KLAposition = [k for k,val in enumerate(AE.get_weights()) if len(val.shape)==0][0]

    AE.compile(Adam(learning_rate=lr,decay=decay),loss='mse')
    history = AE.fit(datasetTrain,epochs=epochs,
                        validation_data=datasetTest,callbacks=[KLAnnealing(KLAposition,AnnealingWeights)])
    
    EncoderContainer.append(Encoder)
    AEContainer.append(AE)
    HistoryContainer.append(history)
    
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

validationNames = [val+'.1.npy' for val in Validation['validation']]
validationNames = list(set(validationNames).intersection(fileNames))
validationPaths = np.array([matrixData+'/'+val for val in validationNames])

datasetVal = tf.data.Dataset.from_generator(DataSequence, 
                                                 args=[validationPaths, batch_size],
                                                 output_types=(tf.float32, tf.float32 ), 
                                                 output_shapes=((None,64,80,1), (None,64,80,1)))

datasetVal = datasetVal.prefetch(2)

valDataFrame = pd.DataFrame()
valDataFrame['ids'] = [val[0:-6] for val in validationNames]

performance = []

fig,axs = plt.subplots(1,5,figsize=(30,15))

for k,block in enumerate(zip(EncoderContainer,AEContainer)):
    
    enc,ae = block

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

completeNames = [val+'.1.npy' for val in Complete['complete']]
completeNames = list(set(completeNames).intersection(fileNames))
completePaths = np.array([matrixData+'/'+val for val in completeNames])

datasetCom = tf.data.Dataset.from_generator(DataSequence, 
                                            args=[completePaths, batch_size],
                                            output_types=(tf.float32, tf.float32 ), 
                                            output_shapes=((None,64,80,1), (None,64,80,1)))

datasetCom = datasetCom.prefetch(2)

completeDataFrame = pd.DataFrame()
completeDataFrame['ids'] = [val[0:-6] for val in completeNames]

for k,block in enumerate(zip(EncoderContainer,AEContainer)):
    
    enc,ae = block
        
    VariationalRepresentation = enc.predict(datasetCom)
    
    completeDataFrame['Dim0_model'+str(k)] = VariationalRepresentation[:,0]
    completeDataFrame['Dim1_model'+str(k)] = VariationalRepresentation[:,1]
    
completeDataFrame.to_csv(outputPath+'/CompDimReduction.csv')

