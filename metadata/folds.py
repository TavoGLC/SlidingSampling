#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
Copyright (c) 2022 Octavio Gonzalez-Lugo 
o use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

###############################################################################
# Loading data
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
metaDataOriginal = pd.read_csv(GlobalDirectory+'/newMetaData.csv')
metaDataOriginal = metaDataOriginal[metaDataOriginal['SimplifiedGEO']=='USA']
metaDataOriginal = metaDataOriginal[metaDataOriginal['geo_long']>-110]

outputpath = r'/media/tavoglc/Datasets/datasets/main/splits'

###############################################################################
# Train Validation splits
###############################################################################

index = np.array(metaDataOriginal['id'])
np.random.shuffle(index)

trainInx,valInx,_,_= train_test_split(index,index,test_size=0.2,random_state=10)

valDF = pd.DataFrame()
valDF['validation'] = valInx
valDF.to_csv(outputpath+'/validation.csv')

###############################################################################
# Train Test Kfolds
###############################################################################

trainDF = pd.DataFrame()
testDF = pd.DataFrame()

folds = KFold(n_splits=5,shuffle=True,random_state=43)

for k,fld in enumerate(folds.split(trainInx)):
    
    train_index, test_index = fld
    
    trainDF['Fold'+str(k)] = trainInx[train_index][0:(trainInx.size//5)*4]
    testDF['Fold'+str(k)] = trainInx[test_index][0:(trainInx.size//5)]

trainDF.to_csv(outputpath+'/train.csv')
testDF.to_csv(outputpath+'/test.csv')

###############################################################################
# Complete metadata 
###############################################################################

complete = metaDataOriginal.query('host_age > 1 & pcr_ct > 0 & geo_alt > 0 & host_race != "unknown" & host_sex != "unknown"')
completeInx = complete['id']

completeDF = pd.DataFrame()
completeDF['complete'] = completeInx

completeDF.to_csv(outputpath+'/complete.csv')
