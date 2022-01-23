# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:59:47 2020

@author: Tommaso
"""
import numpy as np

"""
Assume that the file has the following structure (Parnet format):
    x1, x2, x3, ... (Point Cloud 1)
    y1, y2, y3, ... (Point Cloud 1)

    x1, x2, x3, ... (Point Cloud 2)
    y1, y2, y3, ... (Point Cloud 2)

    ...
"""    
def loadFlatDataset(PATH,p,skiprows=0):
    dataset = np.loadtxt(PATH,dtype=np.double,delimiter=',',skiprows=skiprows)
    
    nrows,ncols = dataset.shape[0],dataset.shape[1]
    nrowsflat,ncolsflat = int(nrows/p),int(ncols*p)
    
    flatdataset = dataset.reshape(nrowsflat,ncolsflat)
    
    return flatdataset,dataset
    
    
    