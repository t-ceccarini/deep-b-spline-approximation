# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:24:54 2020

@author: Tommaso
"""
import torch
import numpy as np
from numpy import linalg as LA

torch.set_default_dtype(torch.float64)

"""
Compute parametrization with spacing proportional to the distances of points with Pytorch and with batches. 
The method used depends on the value of a, e.g:
    if a=0.5 compute centripetal parametrization
    if a=1.0 compute chord length parametrization
"""  
def computeParametrization(curves,a=0.5,device="cuda"):
    batchSize = curves.shape[0]
    l = curves.shape[1]
    #p = curves.shape[2]
    
    t = torch.zeros(batchSize,l).to(device)
    
    dcurves = curves[:,1:] - curves[:,:-1]
    dist = torch.pow(torch.norm(dcurves,dim=2),a)
    t[:,1:] = torch.cumsum(dist,dim=1)
    total = t[:,-1].view(-1,1)
    t = torch.div(t,total)
    
    return t

computeCentripetalParametrization = lambda curves:computeParametrization(curves,a=0.5,device="cpu")
computeChordLengthParametrization = lambda curves:computeParametrization(curves,a=1.0,device="cpu")

"""
Compute uniform parametrization with Pytorch and with batches
"""  
def computeUniformParametrization(curves):
    batchSize = curves.shape[0]
    l = curves.shape[1]
    
    t = torch.linspace(0,1,l)
    tt = t.repeat(batchSize,1)
    
    return tt

"""
Compute parametrization with spacing proportional to the distances of points with Numpy without batches. 
The method used depends on the value of a, e.g:
    if a=0.5 compute centripetal parametrization
    if a=1.0 compute chord length parametrization
"""    
def computeSpacingProportionalParametrization(pts,a=0.5):
    t = np.zeros(pts.shape[0])
    dpts = pts[1:]-pts[:-1]
    dist = np.power((LA.norm(dpts,axis=1)),a)
    t[1:] = np.cumsum(dist)
    
    return t/t[-1] 

def computeUniform(pts):
    l = pts.shape[0]
    
    return np.linspace(0,1,l)
