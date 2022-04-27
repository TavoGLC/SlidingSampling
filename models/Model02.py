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

from joblib import dump
from sklearn import preprocessing as pr

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D,  Dense, Layer, BatchNormalization
from tensorflow.keras.layers import Flatten, Reshape, LayerNormalization, GlobalAveragePooling1D

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
    

class Patches(Layer):
    '''
    Taken from
    https://keras.io/examples/vision/mlp_image_classification/
    '''
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def get_config(self):
        cfg = super().get_config()
        return cfg  
    
    @tf.autograph.experimental.do_not_convert
    def call(self, images,**kwargs):
        
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


class MLPMixerLayer(Layer):
    '''
    Taken from
    https://keras.io/examples/vision/mlp_image_classification/
    '''
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                Dense(units=num_patches),
                BatchNormalization(),
                tfa.layers.GELU(approximate=True),
                BatchNormalization(),
                Dense(units=num_patches),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                Dense(units=num_patches),
                BatchNormalization(),
                tfa.layers.GELU(approximate=True),
                Dense(units=hidden_units),
                BatchNormalization(),
            ]
        )
        self.normalize = LayerNormalization(epsilon=1e-6)
    
    def get_config(self):
        cfg = super().get_config()
        return cfg  
    
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs,**kwargs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

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
    
    Units=[np.product(InputShape),Latent]
    
    if UpSampling:
        finalUnits=Units[::-1]
        InputFunction=Input(shape=(Latent,))
        X=Dense(finalUnits[0],use_bias=False)(InputFunction)
    
    else:
        finalUnits=Units
        InputFunction=Input(shape=InputShape)
        X=Flatten()(InputFunction)
        X=Dense(finalUnits[0],use_bias=False)(X)
                
    
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=Dense(finalUnits[1],use_bias=False)(X)
    X=BatchNormalization()(X)
    
    if UpSampling:
        X=Activation('relu')(X)
        Output=Reshape(InputShape)(X)
    else:
        Output=Activation('relu')(X)
        
    Bottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,Bottleneck

def MakeMixerBlock(inputs,blocks,patch_size,num_patches,embedding_dim,dropout_rate):
    '''
    Parameters
    ----------
    inputs : keras layer
        Input of the mixer block.
    blocks : keras sequential model
        mixer blocks.
    patch_size : int
        size of the image patch, same for each dimention.
    num_patches : int
        number of patches per image.
    embedding_dim : int
        size of the embedding dimention in the mixer block.
    dropout_rate : float
        droput rate in the mixer block.

    Returns
    -------
    representation : keras layer 
        DESCRIPTION.

    '''
    
    patches = Patches(patch_size, num_patches)(inputs)
    x = Dense(units=embedding_dim,use_bias=False)(patches)
    x = blocks(x)
    x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    reshapeDim = np.sqrt(embedding_dim).astype(int)
    representation = Reshape((reshapeDim,reshapeDim,1))(x)
    
    return representation

def MakeMixerCoder(InputShape,Units,NumBlocks,DropoutRate=0.2,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the network.
    Units : array-like
        Contains the dimentionality of the embedding dimentions.
    NumBlocks : int
        Number of mixer blocks.
    DropoutRate : float, optional
        Dropout rate of the mixer block. The default is 0.2.
    PatchSize : int, optional
        size of the segmented patch in the image. The default is 4.
    UpSampling : bool, optional
        Controls the upsamplig or downsampling behaviour of the network.
        The default is False.

    Returns
    -------
    InputFunction : Keras functional model input
        input of the network.
    localCoder : Keras functional model
        Coder model, main body of the autoencoder.

    '''
    
    if UpSampling:
        EmbeddingDimentions=Units[::-1]
    else:
        EmbeddingDimentions=Units
        
    currentSize = np.sqrt(EmbeddingDimentions[0]).astype(int)
    PatchSize = currentSize//2
    num_patches = (currentSize//PatchSize)**2
    
    InputFunction = Input(shape = InputShape)
    X = SpatialAttention(3)(InputFunction)
    X = BatchNormalization()(X)
    MBlocks = keras.Sequential(
        [MLPMixerLayer(num_patches, EmbeddingDimentions[0], DropoutRate) for _ in range(NumBlocks)]
        )
    
    X = MakeMixerBlock(X,MBlocks,PatchSize,num_patches,EmbeddingDimentions[0],DropoutRate)

    for k in range(1,len(EmbeddingDimentions)):
        
        currentSize = np.sqrt(EmbeddingDimentions[k-1]).astype(int)
        PatchSize = currentSize//2
        num_patches = (currentSize//PatchSize)**2
        
        X = SpatialAttention(3)(X)
        X = BatchNormalization()(X)
        MBlocks =  keras.Sequential(
            [MLPMixerLayer(num_patches, EmbeddingDimentions[k], DropoutRate) for _ in range(NumBlocks)]
            )
        X = MakeMixerBlock(X,MBlocks,PatchSize,num_patches,EmbeddingDimentions[k],DropoutRate)
        

    if UpSampling:
        Output = Activation('sigmoid')(X)
        localCoder = Model(inputs=InputFunction,outputs=Output)
        
    else:
        localCoder = Model(inputs=InputFunction,outputs=X)
    
    return InputFunction,localCoder

#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeMixerAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,Encoder=MakeMixerCoder(InputShape,Units,BlockSize)
    #Encoder.summary()
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=MakeMixerCoder(BottleneckInputShape,Units,BlockSize,UpSampling=True)
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeMixerVariationalAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeMixerAutoencoder(InputShape,Units,BlockSize)
    
    InputVAE,VAE=MakeVariationalNetwork(2)
    VAEencoderOutput=VAE(ConvEncoder(InputEncoder))
    ConvVAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=ConvDecoder(ConvVAEencoder(InputEncoder))
    ConvVAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return InputEncoder,InputDecoder,ConvVAEencoder,ConvDecoder,ConvVAEAE

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

DataPath = r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
DataDir = DataPath+'/KmerDataExt.csv'

outputPath = r'/media/tavoglc/Datasets/datasets/main/models/Model02'

###############################################################################
# Data selection
###############################################################################

KmerData = pd.read_csv(DataDir)
KmerData['id'] = [val[0:-2] for val in KmerData['id']]
KmerData = KmerData.set_index('id')

###############################################################################
# Network hyperparameters
###############################################################################

Arch = [37**2,(37//2)**2,(37//4)**2,(37//8)**2]
    
lr = 0.00075
minlr = 0.000001  
epochs = 70 
batch_size = 256
decay=2*(lr-minlr)/epochs
inShape = (37,37,1)
sh = 0.00001
AnnealingWeights = MakeAnnealingWeights(epochs,4,scale=sh)

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
    
    trainData = np.array([np.array(list(val)+[0,0,0,0,0]).reshape((37,37)) for val in trainData]).reshape((-1,37,37,1))
    testData = np.array([np.array(list(val)+[0,0,0,0,0]).reshape((37,37)) for val in testData]).reshape((-1,37,37,1))
    

    datasetTrain = tf.data.Dataset.from_tensor_slices((trainData,trainData))
    datasetTest = tf.data.Dataset.from_tensor_slices((testData,testData))
    
    datasetTrain = datasetTrain.shuffle(buffer_size=100,seed=125)
    datasetTrain = datasetTrain.batch(batch_size)
    datasetTrain = datasetTrain.prefetch(2)
    
    datasetTest = datasetTest.shuffle(buffer_size=100,seed=125)
    datasetTest = datasetTest.batch(batch_size)
    datasetTest = datasetTest.prefetch(2)
    
    _,_,Encoder,Decoder,AE = MakeMixerVariationalAutoencoder(inShape,Arch,1)
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
    valData = np.array([np.array(list(val)+[0,0,0,0,0]).reshape((37,37)) for val in valData]).reshape((-1,37,37,1))
    
    datasetVal = tf.data.Dataset.from_tensor_slices((valData,valData))
    datasetVal = datasetVal.batch(batch_size)
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
    comData = np.array([np.array(list(val)+[0,0,0,0,0]).reshape((37,37)) for val in comData]).reshape((-1,37,37,1))
    datasetCom = tf.data.Dataset.from_tensor_slices((comData,comData))
    
    datasetCom = datasetCom.batch(batch_size)
    datasetCom = datasetCom.prefetch(2)
    
    VariationalRepresentation = enc.predict(datasetCom)
    
    completeDataFrame['Dim0_model'+str(k)] = VariationalRepresentation[:,0]
    completeDataFrame['Dim1_model'+str(k)] = VariationalRepresentation[:,1]
    
completeDataFrame.to_csv(outputPath+'/CompDimReduction.csv')

