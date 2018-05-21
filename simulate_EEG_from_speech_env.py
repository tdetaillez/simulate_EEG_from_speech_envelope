#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:03:51 2017

@author: tobias
"""

import scipy.io as io
import numpy as np

from keras.layers import Conv2DTranspose
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from scipy.stats.mstats import zscore
from keras.models import Sequential
#import hickle as hkl
#import acoustics as ac
import h5py

import sys



def createModel(num_hidden_layers,overWrite=0):
    if overWrite==0:
        streamLength=trainPredWin-tempContext+1
    else:
        streamLength=predictLength
    model = Sequential()
    model.add(Conv2DTranspose(n_hidden**(1),kernel_size=(1,1),strides=(1,1),activation='tanh',input_shape=(streamLength,1,numSides),data_format='channels_last',use_bias=False))
    for lay in range(num_hidden_layers-2,0,-1):
        model.add(Conv2DTranspose(n_hidden**(num_hidden_layers-lay),activation='tanh',kernel_size=(1,1),strides=(1,1),use_bias=False))
    model.add(Conv2DTranspose(numChan,kernel_size=(tempContext,1),activation='tanh',strides=(1,1),use_bias=False))
    model.compile('Nadam','logcosh')
#    model.summary()
        
    return model


#Define Cost function for numpy
def np_logcosh(y_true,y_pred):
    return np.mean(np.log(np.cosh(y_true-y_pred)))




chans=list(range(84))
    
fs=64


clipVal=1.65


numChan=len(chans)
numSides=2 #to attend to
shift=32
gold=True #Gold is our HPC cluster. So these checks aim to make the code run multiple times in parallel on it with different settings

usedLowpass=int(fs/2)
if gold:
    workingDir='/users/asap/kozo6520/MeineForschung/data/'+str(fs)+'hzFS'+str(usedLowpass)+'lowpass'+str(1)+'highpass/'    #GOLD


else:
    workingDir='/home/tobias/gaia/EEG/EEG Bojana/'+str(fs)+'hzFS'+str(usedLowpass)+'lowpass'+str(1)+'highpass/'


##DNN Parameters
n_hidden=4 #4 Neurons for first hidden layer.4^2 for second 4^3 for third hidden layer
num_hidden_layers=3
tempContext=60 #312ms at 64hz
inputContext=10
trainPredWin=60*fs #Training prediction Window
numBlocks=int(50*fs*60/trainPredWin)
blockCount=int(numBlocks/2) #10 repetitions for leave one out evaluation

analysisWindows=[fs*30,fs*10,fs*5,fs*3,fs*2,fs*1,int(fs*0.5),int(fs*0.25),int(fs*0.1)]
tempContext=int(500/1000*fs)
inputContextRange=[1]

pati =        ['01','05','06','08','09','10','11','12','13','14','15','16','17','18','19','20']
attention= [1   ,1   ,2   ,2   ,1   ,2   ,1   ,2   ,2   ,1   ,2   ,1   ,2   ,1   ,2   ,1   ]
numParticipants=len(pati)


#Make calculations as mini-batch for calculation on HPC cluster
thatList=list()
for stream in range(2):
    for leaveBlock in range(25):
        for rep in range(10):
            thatList.append((leaveBlock,rep,stream))
if gold:
    (leaveBlock,rep,stream)=thatList[int(sys.argv[1])]
else:
    (leaveBlock,rep,stream)=thatList[18]
    


eegData=np.zeros((numParticipants,numBlocks,trainPredWin,numChan),dtype=np.float32)
targetAudio=np.zeros((numParticipants,numBlocks,trainPredWin,numSides),dtype=np.float32)

##Read in dataSets
for participant in range(numParticipants):
    transfer = h5py.File(workingDir+'trans_M2P_Patient'+pati[participant]+'.h5','r')
    
    '''Split the dataset into blocks of 1 minute length'''    
    for block in range(numBlocks):
        eegData[participant,block,:,:]=zscore(transfer['data'][chans,block*trainPredWin:(block+1)*trainPredWin].T,axis=0)/3
        if stream==0: # This is the normal case. attended envelope is saved in the 0-entry of targetAudio, unattended envelope is in the 1-entry
            targetAudio[participant,block,:,0]=transfer["EnvA"][block*trainPredWin:(block+1)*trainPredWin,0]
            targetAudio[participant,block,:,1]=transfer["EnvU"][block*trainPredWin:(block+1)*trainPredWin,0]
        elif stream==1: #This is the sanity check. The envelope saved in 0-entry is also from the attended corpus but from a different timestamp; Analog for unattended 
            
            blockalt=np.copy(block)
            while blockalt==block:
                block=np.random.randint(numBlocks)
            targetAudio[participant,blockalt,:,0]=transfer["EnvA"][block*trainPredWin:(block+1)*trainPredWin,0]
            
            while blockalt==block:
                block=np.random.randint(numBlocks)
            targetAudio[participant,blockalt,:,1]=transfer["EnvU"][block*trainPredWin:(block+1)*trainPredWin,0]


'''Design training Set'''
i=0        
trainingDataI=np.zeros(((numBlocks-2)*numParticipants,trainPredWin,numChan)) #2 minutes excluded
trainingDataO=np.zeros(((numBlocks-2)*numParticipants,trainPredWin-tempContext+1,numSides))
for part in range(numParticipants):
    for block in range(numBlocks):
        if ((leaveBlock*2)==block or (leaveBlock*2+1)==(block)):
            continue
        trainingDataI[i,:,:]=eegData[part,block,:,:]
        trainingDataO[i,:,:]=targetAudio[part,block,:-tempContext+1,:]
        i+=1




'''Design Development Set'''
develDataI=np.zeros((numParticipants,trainPredWin,numChan)) 
develDataO=np.zeros((numParticipants,trainPredWin-tempContext+1,numSides))
for part in range(numParticipants):
    develDataI[part,:,:]=eegData[part,leaveBlock*2,:,:]
    develDataO[part,:,:]=targetAudio[part,leaveBlock*2,:-tempContext+1,:]

'''Design Test Set'''
testDataI=np.zeros((numParticipants,trainPredWin,numChan)) 
testDataO=np.zeros((numParticipants,trainPredWin-tempContext+1,numSides))

for part in range(numParticipants):
    testDataI[part,:,:]=eegData[part,leaveBlock*2+1,:,:]
    testDataO[part,:,:]=targetAudio[part,leaveBlock*2+1,:-tempContext+1,:]

revModel=createModel(num_hidden_layers)
tempModelName=workingDir+'../model/RevSingle_part_'+str(rep)+'_'+'att_'+str(leaveBlock)+'block_'+str(num_hidden_layers)+'nonlin_'+'fs'+str(fs)+'.hdf5'


##Early Stopping to get to the point in which the loss on the Development set does not decrease anymore
checkLow = ModelCheckpoint(filepath=tempModelName, verbose=0, save_best_only=True,mode='min',monitor='val_loss')            
early = EarlyStopping(monitor='val_loss',patience=10, mode='min')
'''Training of model'''
revModel.fit(trainingDataO[:,:,None,:],trainingDataI[:,:,None,:],batch_size=2,epochs=3000,verbose=0,callbacks=[early,checkLow],validation_data=(develDataO[:,:,None,:],develDataI[:,:,None,:]))
revModel.load_weights(tempModelName)

'''Simulate EEG for the case that the indeed attended envelope is fed in the "attended" channel'''
testO=np.copy(testDataO)*0
testO[:,:,:2]=testDataO[:,:,:2]
predictionA=revModel.predict(testO[:,:,None,:])[:,:,0,:]


'''predict the test Set, but only one of the input streams at a time. For later spectral analysis'''
finalPredictData=np.copy(testDataO)*0
finalPredictData[:,:,0]=testDataO[:,:,0]
finalPredictA=revModel.predict(finalPredictData[1:3,:,None,:])[:,:,0,:]        

finalPredictData=np.copy(testDataO)*0
finalPredictData[:,:,1]=testDataO[:,:,1]
finalPredictU=revModel.predict(finalPredictData[1:3,:,None,:])[:,:,0,:]   

'''Simulate EEG for the case that the indeed attended envelope is fed in the "unattended" channel'''
switchedInput=np.copy(testDataO)*0
switchedInput[:,:,0]=testO[:,:,1]
switchedInput[:,:,1]=testO[:,:,0]
predictionU=revModel.predict(switchedInput[:,:,None,:])[:,:,0,:]


''' Calculate auditory attention decoding by comparison of predicted EEG in botch cases'''    
corrA=dict()
corrU=dict()
results=np.zeros((len(analysisWindows),4))*np.nan
for win in range(len(analysisWindows)):
    aWindow=analysisWindows[win]
    corrA[str(aWindow)]=np.zeros((numParticipants,int((trainPredWin-aWindow)/shift),1))*np.nan
    corrU[str(aWindow)]=np.zeros((numParticipants,int((trainPredWin-aWindow)/shift),1))*np.nan
    for part in range(numParticipants):
        for samp in range(0,int((trainPredWin-aWindow)/shift)):
            corrA[str(aWindow)][part,samp,0]=np_logcosh(testDataI[part,samp*shift:samp*shift+aWindow,chans],predictionA[part,samp*shift:samp*shift+aWindow,chans])
            corrU[str(aWindow)][part,samp,0]=np_logcosh(testDataI[part,samp*shift:samp*shift+aWindow,chans],predictionU[part,samp*shift:samp*shift+aWindow,chans])
    results[win,0]=np.mean(corrA[str(aWindow)]<corrU[str(aWindow)])
    results[win,1]=np.mean(corrA[str(aWindow)])
    results[win,2]=np.mean(corrU[str(aWindow)])
    results[win,3]=revModel.count_params()


'''Predict impulse response'''
predictLength=tempContext*8+1
toPredictClick=np.zeros((2,predictLength,numSides),dtype=np.float32)
revModel=createModel(num_hidden_layers,1)
revModel.load_weights(tempModelName)

toPredictClick=np.zeros((4,predictLength,1,numSides))    
#Click on Attended Envelope
toPredictClick[0,tempContext*2,0,0]=1    
#Click on Unattended Envelope
toPredictClick[1,tempContext*2,0,1]=1    


#onset and offset answer Attended
toPredictClick[2,tempContext*2:tempContext*4,0,0]=np.ones((tempContext*2))
#onset and offset answer Unattended
toPredictClick[3,tempContext*2:tempContext*4,0,1]=np.ones((tempContext*2))
#onset and offset answer Unrelated


envokedClickCorrect=np.squeeze(revModel.predict((toPredictClick*2-1)))

io.savemat(workingDir+'../resultsDual/RevMulti_'+str(stream)+'_'+str(rep)+'block_'+str(leaveBlock)+'num_hidden_layers'+str(num_hidden_layers)+str(fs)+'Hz.mat',{'finalPredictU':finalPredictU,'finalPredictA':finalPredictA,'envokedClickCorrect':np.squeeze(envokedClickCorrect),'toPredictClick':np.squeeze(toPredictClick),'results':results,'inputContext':inputContext,'tempContext':tempContext,'analysisWindows':analysisWindows})
 




    
    
    
    
    
