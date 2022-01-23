# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:02:40 2020

@author: Tommaso
"""

import torch
from torch import nn
from deep_b_spline_approximation.BSpline import NDPWithBatch3,computeControlPointsWithBatch2

torch.set_default_dtype(torch.float64)


"""KPN-MLP"""

class KnotPlacementNetwork(nn.Module):
        
    def __init__(self,inputSize=300,hiddenSize=500,p=2,k=3,device="cuda"):
        super(KnotPlacementNetwork,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.p = p
        self.d = self.p+1                                
        self.l = int(self.inputSize/self.d)
        self.outputSize = 1
        self.k = k
        self.device = device
        
        #Fully connected layers
        self.inputLayer = torch.nn.Linear(self.inputSize,self.hiddenSize)
        self.hiddenLayer1 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.hiddenLayer2 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.hiddenLayer3 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.outputLayer = torch.nn.Linear(self.hiddenSize,self.outputSize)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dropout3 = torch.nn.Dropout(p=0.2)
        
    def forward(self,x):
        
        batchSize = x.shape[0]
        
        flatcurves = x[:,:self.inputSize-self.l].to(self.device)
        t = x[:,self.inputSize-self.l:].to(self.device)
        
        curves = torch.zeros((batchSize,self.l,self.p))
        curves = flatcurves.reshape(batchSize,self.p,self.l).permute(0,2,1)
        
        x = self.inputLayer(x)
        #x = self.batchNormInput(x)
        #x = self.layerNormInput(x)
        x = self.relu(x)
        
        #print(f"valore di x minimo e massimo dopo il primo layer {torch.min(x)},{torch.max(x)}")
        
        x = self.hiddenLayer1(x)
        #x = self.batchNorm1(x)
        #x = self.layerNorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        #print(f"valore di x minimo e massimo dopo il secondo layer {torch.min(x)},{torch.max(x)}")
        
        x = self.hiddenLayer2(x)
        #x = self.batchNorm2(x)
        #x = self.layerNorm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        #print(f"valore di x minimo e massimo dopo il terzo layer {torch.min(x)},{torch.max(x)}")
        
        x = self.hiddenLayer3(x)
        #x = self.batchNorm3(x)
        #x = self.layerNorm3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        #print(f"valore di x minimo e massimo dopo il quarto layer {torch.min(x)},{torch.max(x)}")
        
        x = self.outputLayer(x)
        #x = self.batchNormOutput(x)
        #x = self.layerNormOutput(x)
        x = self.sigmoid(x)
        
    
        """KS1. Threshold layer"""
        
        tmin = t[:,1].view(-1,1)
        tmax = t[:,-2].view(-1,1)
        
        eps = 1e-5
        
        x = ((x >= tmin) & (x <= tmax))*x + (x < tmin)*(tmin+eps) + (x > tmax)*(tmax-eps)
        
        knot = x.clone()
        
        
        """KS2. Approximation Layer"""
        
        zeros = torch.zeros(batchSize,self.k+1).to(self.device)
        ones = torch.ones(batchSize,self.k+1).to(self.device)
        
        x = torch.cat((zeros,x.view(-1,1),ones),axis=1)                        #Make knot vector u=[0,0,0,0,x,1,1,1,1]
        
        A = NDPWithBatch3(t,self.k,x,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],self.p)).to(self.device)
        splines = torch.zeros((batchSize,self.l,self.p)).to(self.device)
        
        c = computeControlPointsWithBatch2(A,curves[:,1:-1],x,t)
        
        splines[:,1:-1] = torch.matmul(A,c)
        
        splines[:,0] = curves[:,0]
        splines[:,-1] = curves[:,-1]
        
        #splines = torch.matmul(A,c)
        
        return splines,t,curves,x,knot
        


    
