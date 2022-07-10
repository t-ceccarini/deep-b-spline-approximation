# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:33:01 2021

@author: Tommaso
"""

import torch
import numpy as np
from torch import nn
from .directedHausdorff import computeDirectedHausdorffDistance4
from bisect import bisect_left
from .BSpline import NDPWithBatch2,computeControlPointsWithBatch2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def computeTrainingSetComplexity(trainingset,batchSize=10000,device="cuda"):
    
    print("Compute Training Set complexity")
    
    ntrain = trainingset.shape[0]
    totalTrainComplexity = torch.zeros(ntrain)
    
    dataloader = DataLoader(trainingset,batch_size=batchSize,shuffle=False,pin_memory=True)
    dataloader = DataLoader(trainingset,batch_size=batchSize,shuffle=False)
    
    for i,data in enumerate(dataloader,0):
        
        print(f"batch n.{i}")
        
        #batch = data.to(device)
        batch = data
        totalBatchComplexity = computeComplexity(batch)[0]
        totalTrainComplexity[i*batchSize:(i+1)*batchSize] = totalBatchComplexity
    
    torch.where(torch.isnan(totalTrainComplexity))
    
    #define the 98 percentile of curvature of training set
    
    iperc = ntrain - int(ntrain*0.98)
    topk = torch.topk(totalTrainComplexity,iperc)
    thresholdPerc = topk[0][-1]
    
    return thresholdPerc

def computeSegmentation(curves,thresholdPerc):
    #curves = [batch size, curves len, curves components]
    
    """
    #define the 98 percentile of curvature of training set
    ntrain = trainingset.shape[0]
    totalTrainComplexity, trainComplexity = computeComplexity(trainingset,parametrization)
    
    torch.where(torch.isnan(totalTrainComplexity))
    
    iperc = ntrain - int(ntrain*0.98)
    topk = torch.topk(totalTrainComplexity,iperc)
    thresholdPerc = topk[0][-1]
    """
    
    ncurves = curves.shape[0]
    l = curves.shape[1]
    p = curves.shape[2]
    
    #compute the segmentation
    #represents segmentation with list of lists of couples
    #each couple list contains the start index and the end index of each segment 
    segments = [[(0,l-1)] for i in range(0,ncurves)]
    
    totalCurvesComplexities, curvesComplexities = computeComplexity(curves)
    
    torch.where(torch.isnan(totalCurvesComplexities))
    
    indicestosplit = torch.where(totalCurvesComplexities > thresholdPerc)[0]
    
    ncurvesToSeg = indicestosplit.shape[0]
    
    #split each curve that is too complex into segments
    for i in range(ncurvesToSeg):
        
        queue = [(0,l-1)]
        indexCurve = indicestosplit[i]
        
        j = 0           #index of current segment to split in the segments list
        
        #split until every segment is not too complex
        while queue:
            
            segmentIndices = queue.pop(0)
            startIndexSegment, endIndexSegment = segmentIndices[0], segmentIndices[1]
            
            #decrease by 1 iff endIndexSegment = l-1 
            endIndexSegment = endIndexSegment - (endIndexSegment == l-1)                    
            
            segmentComplexity = curvesComplexities[indexCurve,endIndexSegment] - curvesComplexities[indexCurve,startIndexSegment]
            
            lseg = endIndexSegment - startIndexSegment + 1
            
            #if the current segment is not too complex move on
            if segmentComplexity <= thresholdPerc or lseg <= 50:
                
                j = j+1         #check the next segment
             
            else:
                
                #find median (dummy)
                #indexMedianSegment = (startIndexSegment+endIndexSegment)//2
                
                #find median (to verify)
                comp = curvesComplexities[indexCurve,startIndexSegment:endIndexSegment].unsqueeze(0)
                indexMedianSegment = computePercentile(comp).item() + startIndexSegment
                
                #split into two segments
                seg2, seg1 = (indexMedianSegment+1,endIndexSegment + (endIndexSegment == l-2)), (startIndexSegment,indexMedianSegment)
                queue.insert(0,seg2)
                queue.insert(0,seg1)
                
                #remove old segment 
                segments[indexCurve].pop(j)
                
                #insert the two new segments
                segments[indexCurve].insert(j,seg2)
                segments[indexCurve].insert(j,seg1)
            
            
    return thresholdPerc,segments,indicestosplit
    

def computePercentile(complexity,percentile=0.5):
    
    #print(complexity)
    
    max = complexity[:,-1].view(-1,1)
    min = complexity[:,0].view(-1,1)
    normalized = torch.div(complexity-min,max-min)
    
    #print(normalized)
    
    diff = torch.abs(normalized-percentile)
    
    #print(diff)
    
    percentileIndex = torch.argmin(diff,axis=1)
    
    #print(args)
    
    return percentileIndex

def computeComplexity(curves):
    #curves = [batch size, curves len, curves components]
    
    #Compute curvatures = [batch size, curves len]
    #curvature = computeCurvature(curves,parametrization)
    #curvature = computeCurvatureOsculatingCircle(curves)
    curvature = computeRadiusOsculatingCircle2(curves)[0]
    
    #Compute distances between all consecutive couples of points
    distConsecutivePoints = computeDistanceBetweenConsecutivePoints(curves)
    
    #distConsecutivePoints = [batchSize, curves len - 1]
    
    c = torch.abs(curvature[:,:-1]) + torch.abs(curvature[:,1:])
    sumConsecutiveCurvatures = c.squeeze()
    
    #sumConsecutiveCurvatures = [bathcSize, curves len - 1]
    
    elementwiseComplexity = sumConsecutiveCurvatures*distConsecutivePoints*torch.tensor(0.5)
    
    #Compute complexity = [batchSize]
    #complexity = torch.sum(sumConsecutiveCurvatures*distConsecutivePoints, axis=1)*torch.tensor(0.5)
    
    complexity = torch.cumsum(elementwiseComplexity,dim=1)
    totalComplexity = complexity[:,-1]
    
    return totalComplexity,complexity


def computeCurvatureOsculatingCircle(curves):
    
    l = curves.shape[1]
    
    """
    curves = curves.permute(0,2,1)
    curves = F.interpolate(curves,size=l+2,mode='linear')
    curves = curves.permute(0,2,1)
    """
    
    dist =  computeDistanceBetweenConsecutivePoints(curves)
    
    a = dist[:,:-1]
    b = dist[:,1:]
    c = computeDistanceBetweenConsecutivePoints(curves,offset=2)
    
    abc = torch.cat((a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)),axis=0)
    sorted, indices = torch.sort(abc,dim=0,descending=True)
    
    a,b,c = sorted[0],sorted[1],sorted[2]
    
    den = a*b*c
    
    #il problema è che qui può venire fuori la radice di un numero negativo
    k = torch.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4
    
    curvature = 4*k/den
    
    return curvature


def computeRadiusOsculatingCircle(curve):
    
    l = curve.shape[0]
    p = curve.shape[1]
    
    a,b,c = curve[:-2],curve[1:-1],curve[2:]
    
    abc = torch.cat((a,b,c),axis=1)
    
    points = abc.reshape(l-2,-1,p)
    
    means = torch.mean(points,axis=1).reshape(l-2,1,p)
    
    devs = points - means
    
    variances = torch.mean(torch.pow(devs,2),axis=1)
    
    N = ((torch.pow(devs[:,:,0],2) - variances[:,0].reshape(-1,1) + torch.pow(devs[:,:,1],2) - variances[:,1].reshape(-1,1))/2).reshape(l-2,-1,1)
    
    sols = torch.bmm(torch.pinverse(devs),N)
    
    r = torch.sqrt(variances[:,0].reshape(-1,1) + torch.pow(sols[:,0],2) + variances[:,1].reshape(-1,1) + torch.pow(sols[:,1],2))
    
    centers = sols + means.reshape(l-2,-1,1)
    
    curvs = 1/r
    
    return curvs,centers,r


def computeRadiusOsculatingCircle2(curves):
    
    batchSize = curves.shape[0]
    l = curves.shape[1]
    p = curves.shape[2]
    
    a,b,c = curves[:,:-2],curves[:,1:-1],curves[:,2:]
    
    abc = torch.cat((a,b,c),axis=2)
    
    points = abc.reshape(batchSize,l-2,-1,p)
    
    means = torch.mean(points,axis=2).reshape(batchSize,l-2,1,p)
    
    devs = points - means
    
    variances = torch.mean(torch.pow(devs,2),axis=2)
    
    #Versione corretta
    devs2, variances2 = devs.view(batchSize*(l-2),-1,p), variances.view(-1,p)
    N = ((torch.pow(devs2[:,:,0],2) - variances2[:,0].reshape(-1,1) + torch.pow(devs2[:,:,1],2) - variances2[:,1].reshape(-1,1))/2).unsqueeze(-1)
    
    devspinv = torch.pinverse(devs2)
    sols = torch.bmm(devspinv,N)
    
    r = torch.sqrt(variances2[:,0].reshape(-1,1) + torch.pow(sols[:,0],2) + variances2[:,1].reshape(-1,1) + torch.pow(sols[:,1],2))
    
    centers = sols + means.reshape(batchSize*(l-2),-1,1)
    
    curvs = 1/r
    
    curvs,centers,r = curvs.reshape(batchSize,l-2,-1),centers.reshape(batchSize,l-2,-1,1),r.reshape(batchSize,l-2,-1)
    
    #curvature for starting and ending points
    
    p0 = curves[:,0] - torch.mean(curves[:,0:2],axis=1)
    p1 = curves[:,1] - torch.mean(curves[:,0:2],axis=1)
    curvsStart = 2/torch.norm(p1-p0,dim=1).reshape(-1,1)
    
    p0 = curves[:,-2] - torch.mean(curves[:,-2:],axis=1)
    p1 = curves[:,-1] - torch.mean(curves[:,-2:],axis=1)
    curvsEnd = 2/torch.norm(p1-p0,dim=1).reshape(-1,1)
    
    osculating = torch.cat((curvsStart,curvs.squeeze(),curvsEnd),axis=1)
    
    return osculating,curvs,centers,r


def computeCircumference(centers,r,device="cuda:0"):
    
    centers = centers.to(device)
    r = r.to(device)
    
    pi = torch.tensor(np.pi).to(device)
    param = torch.linspace(0,2*pi,100).to(device)
    
    #r2 = torch.pow(r,2)
    
    #dev2 = torch.pow(param - center[0],2)
    
    #circumx = torch.sqrt(r2-dev2)+center[1]
    #circumy = -torch.sqrt(r2-dev2)+center[1]
    
    circumsx = (centers[:,0] + r*torch.cos(param)).unsqueeze(-1)
    circumsy = (centers[:,1] + r*torch.sin(param)).unsqueeze(-1)
    
    circums = torch.cat((circumsx,circumsy),axis=2)
    
    return circums


def drawOsculatingCircles(curve,circums,centers):
    
    curvenp = curve.clone().detach().cpu().numpy()
    circumsnp = circums.clone().detach().cpu().numpy()
    centersnp = centers.clone().squeeze().detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    
    ax.axis("equal")
    
    ax.scatter(curvenp[:,0],curvenp[:,1],color = 'k')
    ax.scatter(centersnp[:,0],centersnp[:,1],s=20, facecolors='none', edgecolors='b')
    
    for i in range(circums.shape[0]): 
        
        ax.plot(circumsnp[i,:,0],circumsnp[i,:,1],linestyle = ':',color='b')
    
    plt.show()
    
    return 0

def computeCurvature(curves,parametrization,device="cuda"):
    
    param = parametrization(curves).to(device)
    
    d1curv,d1t = deriveCurves(curves, param)
    d2curv,d2t = deriveCurves(d1curv,d1t)
    
    num1 = d1curv[:,:,0]*d2curv[:,:,-1]-d1curv[:,:,-1]*d2curv[:,:,0]                        #x'*y''-y'*x''
    den1 = torch.pow(d1curv[:,:,0],2) + torch.pow(d1curv[:,:,-1],2)                         #x'^2+y'^2
    den2 = torch.sqrt(den1) 
    
    curvature = torch.abs(num1)/torch.pow(den2,3)
    
    return curvature


def deriveCurves(curves,param,eps=1e-6):
    #param = parametrization(curves)
    
    dcurves = curves[:,1:] - curves[:,:-1]
    dparam = param[:,1:] - param[:,:-1] 
    dparamsum = dparam[:,:-1]+dparam[:,1:]+eps
    
    curvesderivint = (dcurves[:,:-1]/dparamsum.unsqueeze(-1))+(dcurves[:,1:]/dparamsum.unsqueeze(-1))
    
    a = (dcurves[:,0]/dparam[:,0].unsqueeze(-1)).unsqueeze(1)
    b = (dcurves[:,-1]/dparam[:,-1].unsqueeze(-1)).unsqueeze(1)
    
    curvesderiv = torch.cat((a,curvesderivint,b),axis=1)
    
    paramderiv = torch.cat((dparam[:,0].unsqueeze(-1),dparamsum,dparam[:,-1].unsqueeze(-1)),axis=1)
    
    return curvesderiv,paramderiv
    

def computeDistanceBetweenConsecutivePoints(curves,offset=1):
    batchSize = curves.shape[0]
    l = curves.shape[1]
    p = curves.shape[2]
    
    pdist = nn.PairwiseDistance(keepdim=True)
    curves1, curves2 = curves[:,:-offset], curves[:,offset:]
    fltc1, fltc2 = curves1.reshape(-1,p), curves2.reshape(-1,p)
    fltdist = pdist(fltc1,fltc2)
    dist = fltdist.reshape(batchSize,l-offset)
    
    return dist


def computeSampling(curve,segments,l,device="cuda"):
    
    nsegs = len(segments)
    p = curve.shape[1]
    
    curve2 = torch.zeros(nsegs*l,p).to(device)
    
    dict1 = {}
    dict2 = {}
    start = 0
    
    for seg in segments:
        a,b = seg[0],seg[1]
        
        if b-a+1 > l:
            
            newRange, indicesIncluded, indicesExcluded = doSubSampling(seg,l,start)
            dict1[seg] = newRange
            indicesExcluded.sort()
            dict2[seg] = indicesExcluded
            
            c,d = newRange[0],newRange[1]+1
            curve2[c:d] = curve[indicesIncluded]
            
        elif b-a+1 < l:
            
            newRange, seg2, indices = doSuperSampling(seg,l,start,curve)
            dict1[seg] = newRange
            indices.sort()
            dict2[seg] = indices
            
            c,d = newRange[0],newRange[1]+1
            
            curve2[c:d] = seg2
        
        else:
            
            c,d = a,b+1
            dict1[seg] = seg
            dict2[seg] = []
            curve2[c:d] = curve[c:d]
    
        start = start + l  
    
    return curve2,dict1,dict2

def doSubSampling(s,l,start):
    
    a,b = s[0],s[1]            
    n = b+1-l-a                           #number of indices to exclude
    
    interval = np.arange(a+1,b)
    
    newRange = (start,start+l-1)
    indicesExcluded = np.random.choice(interval,n,replace=False).tolist()
    indicesIncluded = set(np.arange(a,b+1)) - set(indicesExcluded)
    indicesIncluded = list(indicesIncluded)
    
    return newRange,indicesIncluded,indicesExcluded
 
def doSuperSampling(s,l,start,curve):
    
    p = curve.shape[1]
    a,b = s[0],s[1] 
    
    n0 = b - a + 1 
    n1 = n0
    seg1 = curve[a:b+1]
    seg2 = torch.zeros(l,p)
    i = 0
    
    oldIndicesAdded, newIndicesAdded = list(), list()
    
    while(n1 < l):
        
        p1, p2 = seg1[i], seg1[i+1]
        
        if i == 0:
            
            seg2[i],seg2[i+1],seg2[i+2] = p1, torch.lerp(p1,p2,0.5), p2
            seg2[i+3:n1+1] = seg1[i+2:].clone()
            
            newIndicesAdded.append(i+1)
            if i+1 in oldIndicesAdded:
                oldIndicesAdded.remove(i+1)
                newIndicesAdded.append(i+2)
            
        else:
            
            seg2[2*i+1],seg2[2*i+2] = torch.lerp(p1,p2,0.5),p2
            seg2[2*i+3:n1+1] = seg1[i+2:].clone()
            
            newIndicesAdded.append(2*i+1)
            if i+1 in oldIndicesAdded:
                oldIndicesAdded.remove(i+1)
                newIndicesAdded.append(2*i+2)  
        
        n1 = n1 + 1
        
        if i < n0-2:
            i = i + 1
        else:
            i=0
            n0 = n1
            seg1 = seg2[:n1].clone()
            oldIndicesAdded = newIndicesAdded 
            newIndicesAdded = list()
                 
    for elem in oldIndicesAdded:
        newIndicesAdded.append(elem+i)
    
    indicesAdded = set(list(map(lambda x:x+start, newIndicesAdded)))
    
    totalIndices = set(list(np.arange(start,start+l)))
    
    indices = list(totalIndices - indicesAdded)
    
    newRange = (start,start+l-1)
    
    return newRange,seg2,indices

def computeSegmentsNormalization(curveSeg,curve,ranges,indices,l):
    
    nsegs = len(ranges)
    p = curve.shape[1]
    
    curve3 = curveSeg.reshape(nsegs,l,p)
    
    max = torch.max(curve3,axis=1)[0].unsqueeze(1).repeat(1,l,1)
    min = torch.min(curve3,axis=1)[0].unsqueeze(1).repeat(1,l,1)
    
    curveSegNormalized = (curve3-min)/(max-min)
    curveSegNormalized = curveSegNormalized.reshape(-1,p)
    
    return curveSegNormalized
    

def reshapeMLP(curve3,curve4,nsegs,param):
    
    curve3 = curve3.reshape(nsegs,-1,1).squeeze(-1)
    
    if nsegs > 1:
        
        paramSegNormalized = param(curve3)[1]
    
    else:
        
        curve4 = curve3.repeat(2,1)
        paramSegNormalized = param(curve4)[1][-1]
        
    return curve3,curve4,paramSegNormalized

def reshapeCNN(curve3,curve4,nsegs,param):
    
    if nsegs > 1:
        
        paramSegNormalized = param(curve3)[1]
             
    else:
        
        curve4 = curve3.repeat(2,1,1)      
        paramSegNormalized = param(curve4)[1][-1]
    
    return curve3,curve4,paramSegNormalized

def reshapeStd(curve3,curve4,nsegs,param):
    
    if nsegs > 1:
        
        paramSegNormalized = param(curve3)
        
    else:
        
        curve4 = curve3.repeat(2,1,1)      
        paramSegNormalized = param(curve4)
    
    return curve3,curve4,paramSegNormalized

def computeSegmentsParametrization(curve,curveSeg,curveSegNormalized,ranges,indices,l,param,ppn_type,device="cuda"):
    
    nsegs = len(ranges)
    p = curve.shape[1]
    
    curve3 = curveSegNormalized.reshape(nsegs,l,p)
    curve4 = curve3.clone()
    
    if ppn_type == "mlp":
        curve3,curve4,paramSegNormalized = reshapeMLP(curve3,curve4,nsegs,param)
    elif ppn_type == "cnn":
        curve3,curve4,paramSegNormalized = reshapeCNN(curve3,curve4,nsegs,param)
    else:
        curve3,curve4,paramSegNormalized = reshapeStd(curve3,curve4,nsegs,param)
    
    curve3 = curve3.reshape(-1,p)
    paramSegNormalized = paramSegNormalized.reshape(-1)
    
    npoints = curve.shape[0]
    
    param = torch.zeros(npoints).to(device)
    
    computeCL = lambda interval : computeChordLengthSegments(curve,interval)
    
    #riscala ogni parametrizzazione in base alle knots iniziali
    
    rngs = list(ranges.keys())
    chordLenghtSegments = torch.tensor(list(map(computeCL,rngs)))
    
    total = computeChordLengthSegments(curve,(0,curve.shape[0]-1))
    
    ratioSegments = chordLenghtSegments.to(device)/total.to(device)
    internalKnots = torch.cumsum(ratioSegments.reshape(-1,1), dim=0)
    
    knots = torch.zeros(nsegs+1).to(device)
    
    knots[1:-1] = internalKnots[:-1].squeeze(-1)
    knots[-1] = 1
    
    paramSegNormalizedReshape = paramSegNormalized.reshape(nsegs,-1)
    knotsReshape = knots.reshape(nsegs+1,-1)
    paramSegNormalizedRescaled = knotsReshape[:-1,:] + (knotsReshape[1:,:] - knotsReshape[:-1,:])*paramSegNormalizedReshape
    
    #evito duplicati agli estremi della riscala, ad esempio t_99 = t_100 etc.
    paramSegNormalizedRescaled[1:,0] = (paramSegNormalizedRescaled[:-1,-1]+paramSegNormalizedRescaled[1:,1])/2
    
    paramSegNormalizedRescaled = paramSegNormalizedRescaled.reshape(-1)
    
    for key,value in ranges.items():
        
        a,b = key[0], key[1]
        c,d = value[0], value[1]
        j = indices[(a,b)]
        
        #If you've done subsampling add the points previously removed
        if b-a+1 > l:
            
            indicesExcluded = set(j)
            indicesIncluded = list(set(range(a,b+1)) - indicesExcluded)
            
            indicesIncluded.sort()
            
            param[indicesIncluded] = paramSegNormalizedRescaled[c:d+1]
            
            findClosestLeft = lambda number : findClosest(indicesIncluded,number,1)
            findClosestRight = lambda number : findClosest(indicesIncluded,number,0)
            
            closestLeft = list(map(findClosestLeft,j))
            closestRight = list(map(findClosestRight,j))
            
            leftcenter = list(zip(closestLeft,j))
            leftright = list(zip(closestLeft,closestRight))
            
            paramClosestLeft = param[closestLeft]
            paramClosestRight = param[closestRight]
            
            chordLenghtLeftCenter = torch.tensor(list(map(computeCL,leftcenter))).to(device)
            chordLenghtLeftRight = torch.tensor(list(map(computeCL,leftright))).to(device)
            
            ratio = chordLenghtLeftCenter/chordLenghtLeftRight
            
            param[j] = paramClosestLeft + (paramClosestRight - paramClosestLeft)*ratio
            
            #print(key)
        
        #If you've done supersampling remove points
        elif b-a+1 < l:
            param[a:b+1] = paramSegNormalizedRescaled[j]
        
        else:
            param[a:b+1] = paramSegNormalizedRescaled[a:b+1]
    
    return param,paramSegNormalizedRescaled,paramSegNormalized,knots


def findClosest(list,number,left):
    
    pos = bisect_left(list,number)
    
    try:
        val = list[pos-left]
    except:
        print(f"list: {list}")
        print(f"number: {number}")
        print(f"list's length': {len(list)}")
        print(f"pos value: {pos}")
        print(f"left value: {left}")
    
    return val


def computeChordLengthSegments(curve,interval):
    
    start,end = interval[0],interval[1]
    
    segment = curve[start:end+1]
    dseg = segment[1:] - segment[:-1]
    dist = torch.norm(dseg,dim=1)
    
    chordlen = torch.sum(dist)
    
    return chordlen


def computeRefinement2(curve,curveSegNormalized,param,paramSegNormalized,knots,ranges,k,nMaxKnots,nTotalKnots,knotpl=None,device="cuda"):
    
    if nMaxKnots < nTotalKnots:
        nMaxKnots = nTotalKnots
    
    ninitialKnots = knots.shape[0] - 2
    nKnotsToAdd = nTotalKnots - ninitialKnots
    
    zeros = torch.zeros(k).to(device)
    ones = torch.ones(k).to(device)
    knots = torch.cat((zeros,knots,ones))
    
    #Memorize the errors 
    errors = torch.zeros(nMaxKnots+1)
    errorsMSE = torch.zeros(nMaxKnots+1)
    
    #Nel vettore delle knot inserisco il parametro più vicino a quello calcolato dalla kpn.
    #Ogni volta che inserisco un parametro lo rimuovo da quelli disponibili.
    #All'inizio tutti gli indici al di fuori del primo e dell'ultimo sono disponibili.
    param2 = param.clone()[2:-2]
    
    rngs = list(ranges.keys())
    rngs2 = rngs.copy()
    
    #Crea un dizionario multiplicity contenente molteplicità dei vari sottosegmenti.
    #Questa struttura dati si rende necessaria per gestire i casi in cui i sottosegmenti non possono essere più duplicati a causa 
    #delle ridotte dimensioni.
    values = [1]*len(rngs2)
    items = list(zip(rngs2,values))
    multiplicity = dict(items)
    
    #Usato per controllare se l'inserimento di un nuovo nodo causa un aumento dell'errore
    prevDHD = np.inf
    
    mserror = nn.MSELoss()
    
    
    while(nKnotsToAdd > 0):
        
        #Esegui una prima approssimazione globale
        A = NDPWithBatch2(param.unsqueeze(0),k,knots,device)
        
        c = computeControlPointsWithBatch2(A,curve.unsqueeze(0))
    
        spline = torch.matmul(A,c).squeeze(0)
        
        del A,c
        
        nintknots = knots.shape[0]-2*(k+1)
        
        #Controlla quale segmento ha distanza di Hausdorff massima
        
        dhd,index1,index2 = computeDirectedHausdorffDistance4(spline.unsqueeze(0),curve.unsqueeze(0),param.unsqueeze(0),knots.unsqueeze(0))
        mse = mserror(curve,spline)
        
        errors[nintknots] = dhd.item()
        errorsMSE[nintknots] = mse.item()
        
        actualDHD = dhd.item()
        
        prevDHD = actualDHD
        
        #Aggiungi un nuovo nodo nel segmento con distanza di Hausdorff massima
        checkVal = lambda subinterval : checkValue(index1,subinterval)
        subsegment = list(filter(checkVal,rngs2))[0]
        
        checkSubSegment = lambda interval : checkSegment(subsegment,interval)
        
        intervalToRefine = list(filter(checkSubSegment,rngs))[0]
        
        a,b = ranges[intervalToRefine]
        
        #segment, paramSegment = curve[a:b+1],param[a:b+1]
        segment, paramSegment = curveSegNormalized[a:b+1],paramSegNormalized[a:b+1]
        input = torch.cat((segment,paramSegment.unsqueeze(-1)),dim=1).unsqueeze(0)
        
        #reshape for the knot placement network with MLP
        input = input.permute(0,2,1)
        input = input.reshape(1,-1)
        #input = input.repeat(2,1)
        
        try:
        
            knot = knotpl(input)[-1]
        
        except:
            print(f"ranges: {rngs}")
            print(f"ranges2: {rngs2}")
            print(f"subsegment: {subsegment}")
            print(f"interval to refine: {intervalToRefine}")
            print(f"a: {a}")
            print(f"b: {b}")
            print(f"input knot pl shape: {input.shape}")
            
        
        knot = knot[-1]
        
        #Determina il sottointervallo dei nodi opportuno
        
        indexKnot1 = 0
        indexmax = rngs2.index(subsegment)
        rngs3 = rngs2[:indexmax]
        for key,value in multiplicity.items():
            if key in rngs3:
                indexKnot1 = indexKnot1 + value
        
        indexKnot2 = indexKnot1 + multiplicity[subsegment]
        
        #Rescale in the correct interval
        u0,u1 = knots[k+indexKnot1],knots[k+indexKnot2]
        
        knot = u0 + knot*(u1-u0)
        
        #Anziché mettere la knot calcolata dalla rete prendi il parametro disponibile più vicino al valore appena calcolato.
        indexClosest = torch.argmin(torch.abs(param2-knot))
        knot = param2[indexClosest]
        
        #Ogni volta che un parametro viene selezionato viene rimosso dai parametri disponibili
        param2 = torch.cat((param2[:indexClosest],param2[indexClosest+1:]))
        
        knots = torch.cat((knots[:k+indexKnot1+1],knot.unsqueeze(0),knots[k+indexKnot1+1:]))
        
        #Controlla se l'inserimento del nuovo nodo è in ordine corretto
        diff = knots[1:]-knots[:-1]
        if torch.where(diff<0)[0].shape[0] != 0:
            
            #Ordino il vettore dei nodi per evitare nodi in ordine scorretto
            knots = torch.sort(knots)[0]
        
        nKnotsToAdd = nKnotsToAdd - 1
        
        #Suddividi il sottosegmento e aggiorna rngs2 con i due nuovi sottosegmenti se non sono troppo corti
        #Altrimenti aggiorna valore di multiplicity per mantenere corrispondenza tra rngs2 e nodi interni
        
        indexClosestToSeg = torch.argmin(torch.abs(param-knot)).item()
        
        subseg1 = (subsegment[0],indexClosestToSeg)
        subseg2 = (indexClosestToSeg+1,subsegment[1])
            
        if (subseg1[1] - subseg1[0] >= 5) and (subseg2[1] - subseg2[0] >= 5):
            
            rngs2.pop(indexmax)
            rngs2.insert(indexmax,subseg2)
            rngs2.insert(indexmax,subseg1)
            
            #rimuovo il sottosegmento anche da multiplicity
            multiplicity.pop(subsegment)
            
            #aggiungo i due nuovi appena creati
            multiplicity.update([(subseg1,1)])
            multiplicity.update([(subseg2,1)])
        
        else:
            
            #quando l'inserimento è avvenuto al di fuori del segmento
            if indexClosestToSeg < subsegment[0]:
                
                checkVal2 = lambda subinterval : checkValue(np.array([indexClosestToSeg]),subinterval)
                subsegment2 = list(filter(checkVal2,rngs2))[0]
                
                multiplicity[subsegment2] = multiplicity[subsegment2] + 1
            
            else:
                
                #se il sottosegmento non viene suddiviso aggiorna la sua molteplicità
                multiplicity[subsegment] = multiplicity[subsegment] + 1
    
    #Esegui l'ultima approssimazione globale con il vettore dei nodi aggiornato
    A = NDPWithBatch2(param.unsqueeze(0),k,knots,device)
        
    c = computeControlPointsWithBatch2(A,curve.unsqueeze(0))
    
    spline = torch.matmul(A,c).squeeze(0)
    
    del A
    
    nintknots = knots.shape[0]-2*(k+1)
    
    dhd,index1,index2 = computeDirectedHausdorffDistance4(spline.unsqueeze(0),curve.unsqueeze(0),param.unsqueeze(0),knots.unsqueeze(0))  
    mse = mserror(curve,spline)
        
    errors[nintknots] = dhd.item()
    errorsMSE[nintknots] = mse.item()
    
    knots_tmp = torch.zeros(2*(k+1)+nMaxKnots)
    knots_tmp[:knots.shape[0]] = knots
        
    return spline,knots_tmp,c,errors,errorsMSE


def checkValue(value,subsegment):
    
    val = value.item()
    suba, subb = subsegment[0], subsegment[1]
    
    if(val >= suba) and (val <= subb):
        
        return True
    else:
        
        return False


def checkSegment(subsegment,segment):
    
    suba, subb = subsegment[0], subsegment[1]
    sega, segb = segment[0], segment[1]
    
    if (suba >= sega) and (subb<=segb):
        
        return True
    
    else:
        
        return False
    

def checkDistanceSegments(curve,spline,rng,metric):
    
    a,b = rng[0],rng[1]+1
    
    c,spl = curve[a:b+1].unsqueeze(0),spline.unsqueeze(0)
    
    try:
        dist = metric(c,spl)[1]
    except:
        print(f"valore del range: {rng}")
        print(f"numero elementi di curve {c.shape}")
        print(f"numero elementi di spline {spl.shape}")
    
    return dist