# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:00:52 2020

@author: Tommaso
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plotCurve(curve,mod='-'):

    curvenp = curve.clone().detach().cpu().numpy()
    px,py = curvenp[:,0], curvenp[:,1]
    
    plt.figure(figsize = (20,10))
    plt.plot(px,py)
    
    #path="spline.pdf"
    #plt.savefig(path)
    
    plt.show()

    return 0    

def plotWithOP(curve,spline,op,markersize=0.8):
    
    cx,cy = curve[:,:,0].detach().cpu().numpy().squeeze(),curve[:,:,1].detach().cpu().numpy().squeeze()
    sx,sy = spline[:,:,0].detach().cpu().numpy().squeeze(),spline[:,:,1].detach().cpu().numpy().squeeze()
    opx,opy = op[:,:,0].detach().cpu().numpy().squeeze(),op[:,:,1].detach().cpu().numpy().squeeze()
    
    fig, ax = plt.subplots()
    #plt.xlim(9.2,11.2)
    #plt.ylim(9.2,11.2)
    
    #ax.plot(cx, cy, 'b:', markersize=markersize,label='Curve')
    ax.scatter(cx,cy,c='b',label='Curve')
    ax.plot(sx, sy, 'r', label='Spline')
    ax.scatter(sx, sy, s=70, facecolors='none', edgecolors='r', label='Spline')
    #ax.plot(opx, opy, 'k:', markersize=markersize, label='Ort. Projections')
    ax.scatter(opx,opy,c='k',label='Ort. Projections')
    
    for i in range(len(cx)):
        
        cxi,cyi = cx[i],cy[i]
        opxi,opyi = opx[i],opy[i]
        plt.plot([cxi,opxi],[cyi,opyi],'k-')
    
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    fig.tight_layout()
    
    plt.show()
    
    return 0

def plotTogether(curve,spline):
    
    curvenp = curve.clone().detach().cpu().numpy()
    splinenp = spline.clone().detach().cpu().numpy()
    
    px,py = curvenp[:,0], curvenp[:,1]
    sx,sy = splinenp[:,0], splinenp[:,1]
    
    fig, ax = plt.subplots()
    
    ax.plot(px, py, 'b', label='Curve')
    ax.plot(sx, sy, 'r', label='Spline')
    
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    
    plt.show()
    
    return 0

def plotTogether3(curve,splinecnn,splinemlp,splinecentr,splinechord):
    
    curvenp = curve.clone().detach().cpu().numpy()
    splinecnnnp = splinecnn.clone().detach().cpu().numpy()
    splinemlpnp = splinemlp.clone().detach().cpu().numpy()
    splinecentrnp = splinecentr.clone().detach().cpu().numpy()
    splinechordnp = splinechord.clone().detach().cpu().numpy()
    
    px,py = curvenp[:,0], curvenp[:,1]
    scnnx,scnny = splinecnnnp[:,0], splinecnnnp[:,1]
    smlpx,smlpy = splinemlpnp[:,0], splinemlpnp[:,1]
    scentrx,scentry = splinecentrnp[:,0], splinecentrnp[:,1]
    schordx,schordy = splinechordnp[:,0], splinechordnp[:,1]
    
    #plt.figure(figsize=(16,8))
    
    fig, ax = plt.subplots()
    
    ax.plot(scnnx, scnny, 'b', label='cnn')
    ax.plot(smlpx, smlpy, 'r', label='mlp')
    ax.plot(scentrx, scentry, 'k', label='centr')
    ax.plot(schordx, schordy, 'g', label='chord')
    ax.plot(px, py, 'o', markersize=2.0, color= 'k')
    
    ax.grid(axis='y')
    
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    
    figure = plt.gcf()

    figure.set_size_inches(16,11)
    
    path="curvevsspline.pdf"
    plt.savefig(path)
    
    plt.show()
    
    return 0

def plotError(errors,lab):
    
    fig, ax = plt.subplots()
    
    nknots = np.arange(5,26)
    
    ax.plot(nknots, errors[5:], 'b', label=lab)
    
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    
    plt.show()
    
    return 0

def plotTogether2(errors1,errors2,errors3,errors4,labels=['cnn','mlp','centr','chord'],typeError = 'DHD media'):
    
    fig, ax = plt.subplots()
    
    nknots = np.arange(5,26)
    
    plt.xticks(nknots)
    plt.xlabel("Numero di nodi interni")
    plt.ylabel(typeError)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.plot(nknots, errors1[5:], 'b', label=labels[0], linewidth=2)
    ax.plot(nknots, errors2[5:], 'r', label=labels[1], linewidth=2)
    ax.plot(nknots, errors3[5:], 'k', label=labels[2], linewidth=2)
    ax.plot(nknots, errors4[5:], 'g', label=labels[3], linewidth=2)
    
    ax.grid(axis='y')
    
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    
    return plt

def plotDyadic(errors1,errors2,errors3,errors4,labels=['cnn','mlp','centr','chord'],typeError = 'DHD media'):
    
    fig, ax = plt.subplots()
    
    nknots = np.array([0,1,3,7,15,31])
    
    plt.xticks(nknots)
    plt.xlabel("Numero di nodi interni")
    plt.ylabel(typeError)
    
    #ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.plot(nknots, errors1, 'b', label=labels[0], linewidth=2)
    ax.plot(nknots, errors2, 'r', label=labels[1], linewidth=2)
    ax.plot(nknots, errors3, 'k', label=labels[2], linewidth=2)
    ax.plot(nknots, errors4, 'g', label=labels[3], linewidth=2)
    
    ax.grid(axis='y')
    
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    
    return plt

def plotAvgError(errors1,labels='unknown'):
    
    fig, ax = plt.subplots()
    
    nknots = np.arange(5,26)
    
    plt.xticks(nknots)
    plt.xlabel("Numero di nodi interni")
    plt.ylabel("DHD media")
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.plot(nknots, errors1[5:], 'b', label=labels[0], linewidth=2)
    
    ax.grid(axis='y')
    
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    
    return plt

def plotTogetherWithError(curve,spline,indexErrorMax1,indexErrorMax2):
    
    curvenp = curve.clone().detach().cpu().numpy()
    splinenp = spline.clone().detach().cpu().numpy()
    
    px,py = curvenp[:,0], curvenp[:,1]
    sx,sy = splinenp[:,0], splinenp[:,1]
    
    fig, ax = plt.subplots()
    
    ax.plot(px, py, 'b', label='Curve')
    ax.plot(sx, sy, 'r', label='Spline')
    ax.plot(px[indexErrorMax1], py[indexErrorMax1], 'k', 'o')
    ax.plot(sy[indexErrorMax1], sy[indexErrorMax1], 'k', 'o')
    ax.plot(px[indexErrorMax2], py[indexErrorMax2], 'r', 'o')
    
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    
    plt.show()
    
    return 0

def plotSegmentation(curve,segments):
    
    curvenp = curve.clone().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color',['r','b','k','y'])
    
    for seg in segments:
        
        a,b = seg[0],seg[1]
        
        segx,segy = curvenp[a:b+1,0],curvenp[a:b+1,1]
        
        #plt.plot(segx,segy, label='$y = {i}x + {i}$'.format(i=i))
        plt.plot(segx,segy,linewidth=3)
    
    #plt.legend(loc='best')
    plt.show()
    
    return 0

def plotSegmentationWithKnots(curve,segments,parametrization,knots,k=3):
    
    curvenp = curve.clone().detach().cpu().numpy()
    paramnp = parametrization.clone().detach().cpu().numpy()
    knotsnp = knots.clone().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color',['r','b','k','y'])
    
    for seg in segments:
        
        a,b = seg[0],seg[1]
        
        segx,segy = curvenp[a:b+1,0],curvenp[a:b+1,1]
        
        #plt.plot(segx,segy, label='$y = {i}x + {i}$'.format(i=i))
        plt.plot(segx,segy,linewidth=3)
    
    #plt.legend(loc='best')
    #plt.show()
    
    nknots = knotsnp.shape[0]
    
    for u in knotsnp[k:nknots-k]:
        
        indexClosest = np.argmin(np.abs(paramnp-u))
        cx,cy = curvenp[indexClosest,0],curvenp[indexClosest,1]
        
        ax.plot(cx,cy,'d')
    
    plt.show()
    
    return 0

def plotSegmentationWithKnotsAndError(curve,spline,segments,parametrization,knots,indexErrorMax1,indexErrorMax2,k=3):
    
    curvenp = curve.clone().detach().cpu().numpy()
    splinenp = spline.clone().detach().cpu().numpy()
    paramnp = parametrization.clone().detach().cpu().numpy()
    knotsnp = knots.clone().detach().cpu().numpy()
    
    px,py = curvenp[:,0], curvenp[:,1]
    sx,sy = splinenp[:,0], splinenp[:,1]

    fig, ax = plt.subplots()
    ax.set_prop_cycle('color',['r','b','k','y'])
    
    for seg in segments:
        
        a,b = seg[0],seg[1]
        
        segx,segy = curvenp[a:b+1,0],curvenp[a:b+1,1]
        seg2x,seg2y = splinenp[a:b+1,0],splinenp[a:b+1,1]
        
        #plt.plot(segx,segy, label='$y = {i}x + {i}$'.format(i=i))
        plt.plot(segx,segy,linewidth=3)
        plt.plot(seg2x,seg2y,linewidth=3)
    
    #plt.legend(loc='best')
    #plt.show()
    
    nknots = knotsnp.shape[0]
    
    for u in knotsnp[k:nknots-k]:
        
        indexClosest = np.argmin(np.abs(paramnp-u))
        cx,cy = curvenp[indexClosest,0],curvenp[indexClosest,1]
        c2x,c2y = splinenp[indexClosest,0],splinenp[indexClosest,1]
        
        ax.plot(cx,cy,'d')
        ax.plot(c2x,c2y,'d')
    
    ax.plot(px[indexErrorMax1], py[indexErrorMax1], 'k', 'o')
    ax.plot(sy[indexErrorMax1], sy[indexErrorMax1], 'k', 'o')
    ax.plot(px[indexErrorMax2], py[indexErrorMax2], 'r', 'o')
    
    plt.show()
    
    return 0

def plotKnots(curve,parametrization,knots,k=3):
    
    curvenp = curve.clone().detach().cpu().numpy()
    px,py = curvenp[:,0], curvenp[:,1]
    
    paramnp = parametrization.clone().detach().cpu().numpy()
    knotsnp = knots.clone().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    
    #plt.figure(1)
    ax.plot(px,py)
    
    nknots = knotsnp.shape[0]
    
    for u in knotsnp[k:nknots-k]:
        
        indexClosest = np.argmin(np.abs(paramnp-u))
        cx,cy = curvenp[indexClosest,0],curvenp[indexClosest,1]
        
        ax.plot(cx,cy,'d')
    
    plt.show()
    
    return 0