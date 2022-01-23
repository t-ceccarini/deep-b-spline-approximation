# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:42:15 2020

@author: Tommaso
"""
import torch

torch.set_default_dtype(torch.float64)

def computeNKTP(curves,parametrization,nintknots,p=3):
    m = curves.shape[1] - 1
    nknots = nintknots + 2*p + 2
    n = nknots - p - 2
    t = parametrization(curves)
    
    d = (m+1)/(n-p+1)
    
    #print(f"d {d}")
    
    j = torch.arange(1,n-p+1)
    i = (j*d).long()
    #print(f"i {i}")
    #i = i[0]
    
    #print(f"i {i}")
    
    alpha = j*d - i
    ti = t[:,i]
    ti_1 = t[:,i-1]
    u = (1-alpha)*ti_1 + alpha*ti
    
    batchSize = curves.shape[0]
    
    zeros = torch.zeros(batchSize,p+1)
    ones = torch.ones(batchSize,p+1)
    
    knots = torch.cat((zeros,u,ones),axis=1)
    
    return knots

def computeNKTP2(curves,t,nintknots,p=3):
    m = curves.shape[1] - 1
    nknots = nintknots + 2*p + 2
    n = nknots - p - 2
    #t = parametrization(curves)
    
    d = (m+1)/(n-p+1)
    
    #print(f"d {d}")
    
    j = torch.arange(1,n-p+1)
    i = (j*d).long()
    #print(f"i {i}")
    #i = i[0]
    
    #print(f"i {i}")
    
    alpha = j*d - i
    ti = t[:,i]
    ti_1 = t[:,i-1]
    u = (1-alpha)*ti_1 + alpha*ti
    
    batchSize = curves.shape[0]
    
    zeros = torch.zeros(batchSize,p+1)
    ones = torch.ones(batchSize,p+1)
    
    knots = torch.cat((zeros,u,ones),axis=1)
    
    return knots