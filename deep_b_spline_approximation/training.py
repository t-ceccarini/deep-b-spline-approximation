# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:09:47 2020

@author: Tommaso
"""

import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import numpy as np
from ppn import PointParametrizationNetwork,PointParametrizationNetworkCNN2,PointParametrizationNetworkLSTM,Seq2SeqCNN,Seq2SeqRNN,EncoderCNN,EncoderRNN,DecoderCNN,AttentionDecoderRNN,trainSeq2Seq
from kpn import KnotPlacementNetwork
from loadfromtxt import loadFlatDataset
from directedHausdorff import computeMeanHausdorffDistance3
from timeit import default_timer as timer


torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def countParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model,optimizer,criterion,trainingset,device,epochs=10,batchSize=50):
    losses = list()
    
    nCurves = trainingset.shape[0]
    
    dataloader = DataLoader(trainingset,batch_size=batchSize,shuffle=True,pin_memory=True)
    
    start = timer()
    
    for e in range(epochs):
    
        """Training"""
    
        model.train()
    
        runningLoss = 0.0
    
        for i,data in enumerate(dataloader,0):
        
            flpt = data.to(device)
        
            optimizer.zero_grad()
            
            output = ()
        
            #pointsRec,t,pt = model(flpt)
            output = model(flpt)
            
            pointsRec,pt = output[0],output[2]
        
            loss = criterion(pt,pointsRec)
            loss.backward()
        
            optimizer.step()
        
            runningLoss += loss.item()
        
            if i%(nCurves/(batchSize*2)) == (nCurves/(batchSize*2))-1:
                print(f"Epoca: {e+1}, Batch da {int(i+1-(nCurves/(batchSize*2)))} a {i}, Loss: {runningLoss/(nCurves/(batchSize*2))}")
                losses.append(runningLoss/(nCurves/(batchSize*2)))
                runningLoss = 0.0  
        
    end = timer()
    tElapsed = end-start
    print(f"Tempo trascorso: {tElapsed}s")
    
    return model,optimizer,losses


hiddenSize = 1000
nCurves  = 1000
p = 2

#Use Parnet training set

PATH = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\parnet_datasets\train.txt"
flatdataset,dataset = loadFlatDataset(PATH,p)

#Use all examples
"""
flatPoints = torch.tensor(flatdataset)
nCurves = flatPoints.shape[0]
"""

#Use a subset of Parnet training set

np.random.seed(52)
indices = np.random.choice(flatdataset.shape[0],nCurves,replace=False)
flatsubset = flatdataset[indices]
flatPoints = torch.tensor(flatsubset)

dim = flatPoints.shape[1]
l = int(dim/p)


#Use PPN-MLP

ppn = PointParametrizationNetwork(dim,hiddenSize,p)
ppn.to(device)
criterionppn = nn.MSELoss()
optimizerppn = optim.Adam(ppn.parameters(),lr=1e-4)


#PPN-MLP training

modelppn,optimizerppn,losses1 = train(ppn,
                                      optimizerppn,
                                      criterionppn,
                                      flatPoints,
                                      device,
                                      epochs=10,
                                      batchSize=50)


#Prepare trainingset/evaluationset for PPN-SEQ2SEQCNN/SEQ2SEQRNN/CNN/RNN

train1 = flatsubset.reshape(p*nCurves,100)
#train1 = flatdataset.reshape(p*nCurves,100)
train2 = train1.reshape(nCurves,p,100)
#train2 = train1.reshape(240000,2,100)
train3 = torch.tensor(train2)
train4 = train3.permute(0,2,1)


#Use PPN-SEQ2SEQCNN
"""
encoder = EncoderCNN(2,64,128,3,7,0.25,device).to(device)
decoder = DecoderCNN(2,1,64,128,3,7,0.25,0,device).to(device)
seq2seqcnn = Seq2SeqCNN(encoder,decoder).to(device)

criterionseq2seqcnn = nn.MSELoss()
optimizerseq2seqcnn = optim.Adam(seq2seqcnn.parameters(),lr=1e-4)
"""

#PPN-SEQ2SEQCNN training
"""
modelppn = trainSeq2Seq(seq2seqcnn, 
                       optimizerseq2seqcnn, 
                       criterionseq2seqcnn, 
                       train4,  
                       device,
                       epochs=2,
                       batchSize=25) 

"""
#Use PPN-SEQ2SEQRNN
"""
ppnseq2seq = Seq2SeqRNN(inputSizeEncoder=2,hiddenSizeEncoder=64,bidirectional=True,inputSizeDecoder=1,hiddenSizeDecoder=20,finalSize=1)
ppnseq2seq.to(device)

criterionseq2seq = nn.MSELoss()
optimizerseq2seq = optim.Adam(ppnseq2seq.parameters(),lr=1e-1)
"""

#PPN-SEQ2SEQRNN training
"""
modelppn = trainSeq2Seq(ppnseq2seq, 
                             optimizerseq2seq, 
                             criterionseq2seq, 
                             train4,  
                             device,
                             epochs=1,
                             batchSize=25) 

"""
#Use PPN-CNN
"""
ppncnn = PointParametrizationNetworkCNN2()
ppncnn.to(device)

criterioncnn = nn.MSELoss()
optimizercnn = optim.Adam(ppncnn.parameters(),lr=1e-3)
"""

#PPN-CNN training
"""
modelppn,optimizercnn,losses3 = train(ppncnn,
                                       optimizercnn,
                                       criterioncnn,
                                       train4,
                                       device,
                                       epochs=10,
                                       batchSize=50)
"""

#Use PPN-RNN
"""
ppnrnn = PointParametrizationNetworkLSTM()
ppnrnn.to(device)

criterionrnn = nn.MSELoss()
optimizerrnn = optim.Adam(ppnrnn.parameters())

"""
#PPN-RNN training
"""
modelppn,optimizerrnn,losses3 = train(ppnrnn,
                                         optimizerrnn,
                                         criterionrnn,
                                         train4,
                                         device,
                                         epochs=3,
                                         batchSize=25)


"""
#Save PPN model
#torch.save({'model_state_dict': modelrnn.state_dict(),'optimizer_state_dict': optimizerrnn.state_dict()}, 'models/ppn_rnn1.pt')


#Load PPN model already trained

#PATHLOAD = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\models\ppn_mlp2.pt"
"""
checkpoint = torch.load(PATHLOAD)
modelppn = PointParametrizationNetwork(dim,hiddenSize,p)
modelppn.to(device)
modelppn.load_state_dict(checkpoint['model_state_dict'])

modelppn.eval()
"""

#Prepare training set for KPN (using PPN-MLP)
"""
dataloader2 = DataLoader(flatPoints,batch_size=480,shuffle=False,pin_memory=True)

with torch.no_grad():
    
    print("Training set")
    
    for i,data in enumerate(dataloader2,0):
        
        print(f"Iterazione n {i+1}")
        
        crvs = data.to(device)
        splines,t,curves = modelppn(crvs)
        
        if i==0:
            flatPointsAndT = torch.cat((crvs,t),axis=1).cpu()
        else:
            newExample = torch.cat((crvs,t),axis=1).cpu()
            flatPointsAndT = torch.cat((flatPointsAndT,newExample),axis=0).cpu()
        
        print(f"flat points and t shape {flatPointsAndT.shape}")
    
"""   
   
#Define and train KPN-MLP
"""
kpnet = KnotPlacementNetwork(inputSize=300,hiddenSize=500,p=2,k=3)
kpnet.to(device)

criterionkp = nn.MSELoss()
optimizerkp = optim.Adam(kpnet.parameters(),lr=1e-4)

modelkp,optimizerkp,losses2 = train(kpnet,
                                    optimizerkp,
                                    criterionkp,
                                    flatPointsAndT,
                                    device,
                                    epochs=10,
                                    batchSize=50)

"""
#Save KPN model
#torch.save({'model_state_dict': modelkp.state_dict(),'optimizer_state_dict': optimizerkp.state_dict()}, 'models/kpn_mlp4.pt')


