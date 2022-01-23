# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:20:32 2020

@author: Tommaso
"""
import torch
import numpy as np
from preprocessing import computeSegmentation,computeSampling,computeSegmentsNormalization,computeSegmentsParametrization,computeRefinement2,computeTrainingSetComplexity
from loadfromtxt import loadFlatDataset
from parametrization import computeParametrization,computeCentripetalParametrization,computeChordLengthParametrization
from ppn import PointParametrizationNetwork,PointParametrizationNetworkCNN2
from kpn import KnotPlacementNetwork

torch.set_default_dtype(torch.float64)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

p = 2
nCurves = 240000
l = 100

#Use Parnet training set

PATH = r"parnet_datasets\train.txt"

flatdataset,dataset = loadFlatDataset(PATH,p)
train1 = flatdataset.reshape(p*nCurves,100)
train2 = train1.reshape(240000,2,100)
train3 = torch.tensor(train2)
trainingset = train3.permute(0,2,1)


PATHEVAL = r"parnet_datasets\evalset1.txt"

flatevalset,evalset = loadFlatDataset(PATHEVAL,p)
flatevaltorch = torch.tensor(flatevalset)

ncurveval = evalset.shape[0] // p
npoints = evalset.shape[1] 

eval1 = torch.tensor(evalset)
eval2 = eval1.reshape(ncurveval,p,npoints)
curves = eval2.permute(0,2,1)


PATHLOADPPN = r"models\ppn_cnn1.pt"
PATHLOADKPN = r"models\kpn_mlp4.pt"

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

thresholdPerc = computeTrainingSetComplexity(trainingset)

thresholdPerc,segments,itosplit = computeSegmentation(curves,thresholdPerc)

curves = curves.to(device)

avghd = list()

totalErrors = torch.zeros(500,26)
totalErrorsMSE = torch.zeros(500,26)

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

    spline,knots,cps,errors,errorsMSE = computeRefinement2(curve,curveSegNormalized,param,paramSegNormalized,knots,ranges,3,25,kpnet,device)
    
    totalErrors[i] = errors
    totalErrorsMSE[i] = errorsMSE
    
    print(f"Errori DHD per curva {i} con 5,10,15,20,25 nodi: {errors[[5,10,15,20,25]]}")
    print(f"Errori MSE per curva {i} con 5,10,15,20,25 nodi: {errorsMSE[[5,10,15,20,25]]}")
    
    
    meanError = torch.mean(totalErrors[:i+1],axis=0)
    meanErrorMSE = torch.mean(totalErrorsMSE[:i+1],axis=0)
    print(f"Errore DHD medio fino alla curva {i} con 5,10,15,20,25 nodi: {meanError[[5,10,15,20,25]]}")
    print(f"Errore MSE medio fino alla curva {i} con 5,10,15,20,25 nodi: {meanErrorMSE[[5,10,15,20,25]]}")
    
    torch.cuda.empty_cache()

