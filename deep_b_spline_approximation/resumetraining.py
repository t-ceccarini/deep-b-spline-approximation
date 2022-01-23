# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:04:11 2020

@author: Tommaso
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn,randn,optim
from torch.utils.data import TensorDataset,DataLoader
from ppn import PointParametrizationNetwork
from loadfromtxt import loadFlatDataset
from directedHausdorff import computeMeanHausdorffDistance3
from timeit import default_timer as timer

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

hiddenSize = 1000
nCurves  = 5000
p = 2
batchSize=5

PATH = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\parnet_datasets\train.txt"
flatdataset,dataset = loadFlatDataset(PATH,p)

#select a subset of training set
np.random.seed(52)
indices = np.random.choice(flatdataset.shape[0],nCurves,replace=False)
flatsubset = flatdataset[indices]
flatPoints = torch.tensor(flatsubset)
dim = flatPoints.shape[1]
datasetLoader = DataLoader(flatPoints,batch_size=batchSize,shuffle=True)

PATHLOAD = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\models\resume3.pt"

ppn = PointParametrizationNetwork(dim,hiddenSize,p)
ppn.to(device)
optimizer = optim.Adam(ppn.parameters(),lr=1e-4)

checkpoint = torch.load(PATHLOAD)
ppn.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

criterion = nn.MSELoss()

losses = list()
avghd = list()

PATHEVAL = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\parnet_datasets\evalset2.txt"
flatevalset,evalset = loadFlatDataset(PATHEVAL,p)
flatevaltorch = torch.tensor(flatevalset)

start = timer()

for e in np.arange(epoch,epoch+10):
    
    runningLoss = 0.0
    for i,data in enumerate(datasetLoader,0):
        #flpt = data[0]
        flpt = data.to(device)
        #pt = flpt.reshape(l,p)
        #pt = flpt.reshape(l,p).cuda()
        
        optimizer.zero_grad()
        
        pointsRec,t,pt = ppn(flpt)
        #pointsRec,t = ppngpu(flpt)
        
        loss = criterion(pt,pointsRec)
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(ppn.parameters(),5.0)
        optimizer.step()
        
        runningLoss += loss.item()
        if i%(nCurves/(batchSize*2)) == (nCurves/(batchSize*2))-1:
            print(f"Epoca: {e+1}, Batch da {int(i+1-(nCurves/(batchSize*2)))} a {i}, Loss: {runningLoss/(nCurves/(batchSize*2))}")
            losses.append(runningLoss/(nCurves/(batchSize*2)))
            runningLoss = 0.0
    
    """Evaluation"""
    with torch.no_grad():    
        #model = ppn.cpu()
        avghdppn = computeMeanHausdorffDistance3(flatevaltorch,ppn,3,p)
        print(f"Average Hausdorff distance on validation set: {avghdppn}")
        avghd.append(avghdppn)
        #ppn.to(device)


end = timer()
tElapsed = end-start
print(f"Tempo trascorso: {tElapsed}s")

PATHSAVE = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\models\resume4.pt"
torch.save({'epoch':e+1,'model_state_dict': ppn.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},PATHSAVE)