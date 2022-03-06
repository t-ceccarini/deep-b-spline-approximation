# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:16:09 2020

@author: Tommaso
"""
import torch
import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.spatial.distance import directed_hausdorff
from .BSpline import NDPWithBatch2,computeControlPointsWithBatch2

torch.set_default_dtype(torch.float64)

def computeMeanHausdorffDistance3(dataset,model,k,p):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    dataset = dataset.to(device)
    
    output = ()
    
    output = model(dataset)
    
    #if len(output) == 4:
        #print(output[3])
    
    splines,curves = output[0],output[2]
    
    splinesnp,curvesnp = splines.cpu().detach().numpy(),curves.cpu().detach().numpy()
    n = splinesnp.shape[0]
    
    sumHd = 0.0
    
    for i in np.arange(n):
        spline,curve = splinesnp[i],curvesnp[i]
        
        hd = directed_hausdorff(curve,spline)[0]
        #hd = directed_hausdorff(spline,curve)[0]
        
        sumHd = sumHd +hd
    
    return sumHd/n

def computeDirectedHausdorffDistance4(splines,curves,params,knots,nEval=5000,k=3,device="cuda"):
    
    batchSize = curves.shape[0]
    l = curves.shape[1]
    p = curves.shape[2]
    
    tSampled = np.linspace(0,1,num=nEval)
    
    DHD = np.zeros(batchSize)
    indices1 = np.zeros(batchSize,dtype='int')
    indices2 = np.zeros(batchSize,dtype='int')
    
    for i in range(batchSize):
        
        #print(f"curva n. {i}")
        
        curve = curves[i].clone().detach().cpu().numpy()
        param = params[i].clone().detach().cpu().numpy().squeeze()
        kn = knots[i].clone().detach().cpu().numpy()
        #guess = guesses[i].clone().detach().cpu().numpy().squeeze()
        
        try:
            spline = make_lsq_spline(param,curve,kn)
            spleval = spline(tSampled)
        except:
            curve2 = curves[i].clone()
            param2 = params[i].clone()
            kn2 = knots[i].clone()
            
            A = NDPWithBatch2(param2.unsqueeze(0),k,kn2)
        
            c = computeControlPointsWithBatch2(A,curve2.unsqueeze(0))
    
            spleval = torch.matmul(A,c).squeeze(0).clone().detach().cpu().numpy()
        
        
        dhd,indexC,indexS = directed_hausdorff(curve,spleval)
        
        DHD[i] = dhd
        indices1[i] = indexC
        indices2[i] = indexS
    
    return DHD,indices1,indices2




