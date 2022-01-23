# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:34:28 2021

@author: Tommaso
"""

import torch
import numpy as np
from torch import nn
from loadfromtxt import loadFlatDataset
from preprocessing import computeSegmentation,computeSampling,computeSegmentsNormalization,computeSegmentsParametrization,computeTrainingSetComplexity
from parametrization import computeParametrization, computeCentripetalParametrization, computeChordLengthParametrization
from directedHausdorff import computeDirectedHausdorffDistance4
from ppn import PointParametrizationNetwork,PointParametrizationNetworkCNN2
from kpn import KnotPlacementNetwork
from BSpline import NDPWithBatch2,computeControlPointsWithBatch2

torch.set_default_dtype(torch.float64)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

p = 2
nCurves = 240000
l = 100

#Use Parnet training set

PATH = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\parnet_datasets\train.txt"

flatdataset,dataset = loadFlatDataset(PATH,p)
train1 = flatdataset.reshape(p*nCurves,100)
train2 = train1.reshape(240000,2,100)
train3 = torch.tensor(train2)
trainingset = train3.permute(0,2,1)


PATHEVAL = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\parnet_datasets\evalset4.txt"

print(f"percorso file {PATHEVAL}")

flatevalset,evalset = loadFlatDataset(PATHEVAL,p)
flatevaltorch = torch.tensor(flatevalset)

ncurveval = evalset.shape[0] // p
npoints = evalset.shape[1] 

eval1 = torch.tensor(evalset)
eval2 = eval1.reshape(ncurveval,p,npoints)
curves = eval2.permute(0,2,1)


PATHLOADPPN = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\models\ppn_cnn1.pt"
PATHLOADKPN = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\models\kpn_mlp4.pt"

#Load point parametrization network
dim, hiddenSize = 200,1000

#ppn = PointParametrizationNetwork(dim,hiddenSize,p,device="cpu")
ppn = PointParametrizationNetworkCNN2(device="cpu") 
ppn.to(device)

checkpoint1 = torch.load(PATHLOADPPN)
ppn.load_state_dict(checkpoint1['model_state_dict'])
ppn = ppn.eval()

#Load knot placement network
kpnet = KnotPlacementNetwork(inputSize=300,hiddenSize=500,p=2,k=3,device="cpu")
kpnet.to(device)

checkpoint2 = torch.load(PATHLOADKPN)
kpnet.load_state_dict(checkpoint2['model_state_dict'])
kpnet = kpnet.eval()

parametrization = computeCentripetalParametrization
mserror = nn.MSELoss()

thresholdPerc = computeTrainingSetComplexity(trainingset)

thresholdPerc,segments,itosplit = computeSegmentation(curves,thresholdPerc)

curves = curves.to(device)

avghd = list()

totalErrorsDHD = torch.zeros(500,6)
totalErrorsMSE = torch.zeros(500,6)

#Riproducibilità
torch.manual_seed(84)
np.random.seed(84)

parametrization = ppn
#parametrization = computeCentripetalParametrization

for i,curve in enumerate(curves,0):
    
    print(f"curva {i}")
    
    segs = segments[i]

    l = trainingset.shape[1]

    curveSeg, ranges, indices = computeSampling(curve,segs,l,device)
    curveSegNormalized = computeSegmentsNormalization(curveSeg,curve,ranges,indices,l)

    param,paramSegNormalizedRescaled,paramSegNormalized,knots = computeSegmentsParametrization(curve,curveSeg,curveSegNormalized,ranges,indices,l,parametrization,device)
    
    zeros = torch.tensor([0,0,0],dtype=torch.double)
    ones = torch.tensor([1,1,1],dtype=torch.double)
    
    for j in range(6):
        
        if j == 0:
            dim_j = 2
            
        else:
            dim_j = 2*dim_j - 1
        
        knots_j = torch.linspace(0,1,dim_j,dtype=torch.double)
        
        knots = torch.cat((zeros,knots_j,ones))
        
        A = NDPWithBatch2(param.unsqueeze(0),3,knots,"cpu")
        cp = computeControlPointsWithBatch2(A,curve.unsqueeze(0))
        spline = torch.matmul(A,cp).squeeze(0)
        
        dhd,index1,index2 = computeDirectedHausdorffDistance4(spline.unsqueeze(0),curve.unsqueeze(0),param.unsqueeze(0),knots.unsqueeze(0))  
        mse = mserror(curve,spline)
        
        totalErrorsDHD[i,j] = dhd.item()
        totalErrorsMSE[i,j] = mse.item()
    
    
    print(f"Errori DHD per curva {i} con 0,1,3,7,15,31 nodi interni: {totalErrorsDHD[i]}")
    print(f"Errori MSE per curva {i} con 0,1,3,7,15,31 nodi: {totalErrorsMSE[i]}")
    
    meanErrorDHD = torch.mean(totalErrorsDHD[:i+1],axis=0)
    meanErrorMSE = torch.mean(totalErrorsMSE[:i+1],axis=0)
    
    print(f"Errore DHD medio fino alla curva {i} con 0,1,3,7,15,31 nodi: {meanErrorDHD}")
    print(f"Errore MSE medio fino alla curva {i} con 0,1,3,7,15,31 nodi: {meanErrorMSE}")
    
    torch.cuda.empty_cache()

    