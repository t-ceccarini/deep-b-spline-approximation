# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:10:09 2021

@author: Tommaso
"""

from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
import bezier
import math
from math import pi


def points_on_circumference(center=(0.41, 0.36), r=0.38, n=96):
    return [
        (
            center[0]+(math.cos(2 * pi / n * x) * r),  # x
            center[1] + (math.sin(2 * pi / n * x) * r)  # y

        ) for x in range(0, n)]

def makeAnimationBezier(knots,cps_t,k,path,nFrames=24,offset=0):
    
    t = np.linspace(0,1,100)
    
    for i in range(0,nFrames):
    
        #print(f"i punti di controllo sono {cps}")    
    
        #spl = BSpline(knots,cps_t[i],k)
        
        #spleval = spl(t)
        
        bez = bezier.Curve(np.asfortranarray(cps_t[i].T),k)
        bezeval = bez.evaluate_multi(t).T
        
        num = offset + i
        
        plt.figure()
        
        plt.xlim([-0.1, 1.7])
        plt.ylim([-0.1, 1.1])
        
        plt.axis('off')
        
        plt.plot(cps_t[i,:,0],cps_t[i,:,-1],marker='$\\bigotimes$',markersize=8.0,color='k',alpha=0.8,linewidth=1.0)
        #plt.plot(spleval[:,0],spleval[:,-1],'b',linewidth=2.0)
        plt.plot(bezeval[:,0],bezeval[:,-1],'b',linewidth=2.0)
        
        path2 = path+f"\\Img-{num}.pdf"
        plt.savefig(path2, bbox_inches='tight')
        
        
    #plt.close()
    
    #cps = cps - delta
    
    return cps_t

def makeAnimationBSpline(knots,cps_t,k,delta,path,nFrames=24,offset=0):
    
    t = np.linspace(0,1,100)
    
    for i in range(0,nFrames):
    
        #print(f"i punti di controllo sono {cps}")    
    
        spl = BSpline(knots,cps_t[i],k)
        
        spleval = spl(t)
        
        num = offset + i
        
        plt.figure()
        
        plt.xlim([-0.1, 1.7])
        plt.ylim([-0.1, 1.1])
        
        plt.axis('off')
        
        plt.plot(cps_t[i,:,0],cps_t[i,:,-1],marker='$\\bigotimes$',markersize=8.0,color='k',alpha=0.8,linewidth=1.0)
        plt.plot(spleval[:,0],spleval[:,-1],'b',linewidth=2.0)
        
        path2 = path+f"\\Img-{num}.pdf"
        plt.savefig(path2, bbox_inches='tight')
        
        
    #plt.close()
    
    #cps = cps - delta
    
    return cps_t



#Bezier
k=3
knots = np.array([0,0,0,0,1,1,1,1])

cps_t = np.zeros((96,4,2))

cps_t[0] = np.array([[0,1],[0.27,0],[1.05,0.97],[1.49,0]])

cps_0 = np.array([0,1]).reshape(1,1,2).repeat(96,0)


cps_11 = np.array(points_on_circumference())

cps_1 = np.concatenate([cps_11[67:],cps_11[:67]]).reshape(96,1,2)

cps_2 = np.array([[1.05,0.97]]).reshape(1,1,2).repeat(96,0)
cps_3 = np.array([[1.49,0]]).reshape(1,1,2).repeat(96,0)

cps_t = np.concatenate((cps_0,cps_1,cps_2,cps_3),axis=1)

path = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\bezier"

cps2_t = makeAnimationBezier(knots,cps_t,k,path,nFrames=96)

#B-spline
"""
zeros = np.zeros(3)
ones = np.ones(3)
int_knots = np.linspace(0,1,6)

knots = np.concatenate((zeros,int_knots,ones))

k=3

cps_t[0] = np.array([[0.0,1.0],
                     [0.2,0.0],
                     [0.4,1.0],
                     [0.6,0.0],
                     [0.8,1.0],
                     [1.0,0.0],
                     [1.2,1.0],
                     [1.6,0.0]])




delta = np.array([[0,0],
                  [0,0],
                  [0.0,-1.1/48],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,0]])


for i in np.arange(1,48):
    cps_t[i] = cps_t[i-1]+delta



delta = np.array([[0,0],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0.0,-1.0/48],
                  [0,0]])


for i in np.arange(48,96):
    cps_t[i] = cps_t[i-1]+delta
"""
#path = r"C:\Users\Tommaso\Desktop\Tesi\source\knotplacement\bspline"
"""
cps2_t = makeAnimationBSpline(knots,cps_t,k,delta,path,nFrames=96)
"""