# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:22:59 2020

@author: Tommaso
"""
import torch
from scipy.interpolate import make_lsq_spline

torch.set_default_dtype(torch.float64)

def makeBSplineWithNumpy(curve,param,knots):
    
    tt = param.clone().detach().cpu().numpy()
    curvenp = curve.clone().detach().cpu().numpy()
    
    px = curvenp[:,0]
    py = curvenp[:,-1]
    
    uu = knots.clone().detach().cpu().numpy()
    
    splx = make_lsq_spline(tt,px,uu)
    sply = make_lsq_spline(tt,py,uu)
    
    return splx,sply,tt


def computeBSplineApproximation(curves,t,device,u=None,k=3):
    
    if u == None:
    
        u = torch.cat((torch.tensor([0.0]*(k+1)),torch.tensor([1.0]*(k+1))))
    
    A = NDPWithBatch(t,k,u,device)
        
    c = computeControlPointsWithBatch2(A,curves)
      
    return torch.matmul(A,c)
 
 
def computeControlPointsWithDecomposition(N,p):
    NTranspose = N.T
    a = NTranspose.mm(N)
    b = NTranspose.mm(p)
    
    c,lu = torch.solve(b,a) 
    
    return c
    
def computeControlPoints(N,p):
    NTranspose = N[:,1:-1].T
    a = NTranspose.mm(N[:,1:-1])
    
    """calcolo inversa di N^T*N"""
    
    #senza decomposizione
    inva = a.inverse()
    
    #con decomposizione
    #u = torch.cholesky(a)
    #inva = torch.cholesky_inverse(u)
    
    """calcolo q"""
    pi = p[1:-1]
    
    m_1,d = pi.shape[0], pi.shape[1]
    
    p0,pm = p[0].repeat(m_1,1), p[-1].repeat(m_1,1)
    N0,Nn = N[:,0].view(-1,1).repeat(1,d), N[:,-1].view(-1,1).repeat(1,d)
    
    print(N0.shape)
    print(p0.shape)
    
    qi = pi - N0*p0 - Nn*pm
    
    #sumN = torch.sum(N,axis=0)[1:-1].view(-1,1).repeat(1,d)
    
    #print(sumN.shape)
    print(qi.shape)
    
    q = torch.matmul(N[:,1:-1].T,qi)
    #q = sumN*qi
    
    #print(f"b device {b.device}")
    #print(f"p device {p.device}")
    
    c = torch.matmul(inva,q)
    
    return c

def computeControlPoints2(N,p):
    NTranspose = N[1:-1,1:-1].T
    a = NTranspose.mm(N[1:-1,1:-1])
    
    """calcolo inversa di N^T*N"""
    
    #senza decomposizione
    inva = a.inverse()
    
    #con decomposizione
    #u = torch.cholesky(a)
    #inva = torch.cholesky_inverse(u)
    
    """calcolo q"""
    pi = p[1:-1]
    
    m_1,d = pi.shape[0], pi.shape[1]
    
    p0,pm = p[0].repeat(m_1,1), p[-1].repeat(m_1,1)
    N0,Nn = N[1:-1,0].view(-1,1).repeat(1,d), N[1:-1,-1].view(-1,1).repeat(1,d)
    
    #print(N0.shape)
    #print(p0.shape)
    
    qi = pi - N0*p0 - Nn*pm
    
    #sumN = torch.sum(N,axis=0)[1:-1].view(-1,1).repeat(1,d)
    
    #print(sumN.shape)
    #print(qi.shape)
    
    q = torch.matmul(NTranspose,qi)
    #q = sumN*qi
    
    #print(f"b device {b.device}")
    #print(f"p device {p.device}")
    
    c = torch.matmul(inva,q)
    
    return c

def computeControlPoints3(N,p):
    NTranspose = N[1:-1].T
    a = NTranspose.mm(N[1:-1])
    
    """calcolo inversa di N^T*N"""
    
    #senza decomposizione
    inva = a.inverse()
    
    #con decomposizione
    #u = torch.cholesky(a)
    #inva = torch.cholesky_inverse(u)
    
    """calcolo q"""
    pi = p[1:-1]
    
    m_1,d = pi.shape[0], pi.shape[1]
    
    p0,pm = p[0].repeat(m_1,1), p[-1].repeat(m_1,1)
    N0,Nn = N[1:-1,0].view(-1,1).repeat(1,d), N[1:-1,-1].view(-1,1).repeat(1,d)
    
    #print(N0.shape)
    #print(p0.shape)
    
    qi = pi - N0*p0 - Nn*pm
    
    #sumN = torch.sum(N,axis=0)[1:-1].view(-1,1).repeat(1,d)
    
    #print(sumN.shape)
    #print(qi.shape)
    
    q = torch.matmul(NTranspose,qi)
    #q = sumN*qi
    
    #print(f"b device {b.device}")
    #print(f"p device {p.device}")
    
    c = torch.matmul(inva,q)
    
    return c


def computeControlPointsWithBatch(N,p):
    NTranspose = N[:,:,1:-1].permute(0,2,1)
    a = torch.matmul(NTranspose,N[:,:,1:-1])
    
    #NTranspose = N.permute(0,2,1)
    #a = torch.matmul(NTranspose,N)
    
    """calcolo inversa di N^T*N"""
    inva = torch.inverse(a)
    
    """calcolo q"""
    
    pi = p[:,1:-1]
    #pi = p[:,:]
    
    m_1,d = pi.shape[1], pi.shape[2]
    
    p0,pm = p[:,0].repeat(1,m_1).view(-1,m_1,d), p[:,-1].repeat(1,m_1).view(-1,m_1,d)
    #N0,Nn = N[:,:,0].view(-1,m_1,1).repeat(1,d), N[:,:,-1].view(-1,m_1,1).repeat(1,d)
    #N0,Nn = N[:,:,0].repeat(d,1).view(-1,m_1,d), N[:,:,-1].repeat(d,1).view(-1,m_1,d)
    N0,Nn = N[:,:,0].view(-1,m_1,1).repeat(1,1,d), N[:,:,-1].view(-1,m_1,1).repeat(1,1,d)
    
    #print(N0.shape)
    #print(p0.shape)
    #print(pi.shape)
    
    qi = pi - N0*p0 - Nn*pm
    
    #print(qi.shape)
    
    q = torch.matmul(N[:,:,1:-1].permute(0,2,1),qi)
    # = torch.matmul(N.permute(0,2,1),qi)
    
    #print(q.shape)
    #print(inva.shape)
    
    c = torch.matmul(inva,q)
    
    return c

def computeControlPointsWithBatch2(N,p,u=None,t=None):
    #NTranspose = N[:,:,1:-1].permute(0,2,1)
    #a = torch.matmul(NTranspose,N[:,:,1:-1])
    
    nintknots = N.shape[2]-4
    
    NTranspose = N.permute(0,2,1)
    a = torch.matmul(NTranspose,N)
    
    """calcolo inversa di N^T*N"""
    
    try:
        inva = torch.inverse(a)
    except:
        #u = torch.cholesky(a)
        #inva = torch.cholesky_inverse(u)
        print(f"Calcolo pseudoinversa, matrice non invertibile. Numero di nodi Interni: {nintknots}.")
        inva = torch.pinverse(a)
    
    """calcolo N^T*p"""
    
    #print(NTranspose.shape)
    #print(p.shape)
    
    q = torch.matmul(NTranspose,p)
    
    #print(q.shape)
    #print(inva.shape)
    
    c = torch.matmul(inva,q)
    
    return c


def NDP(t,k,u,device="cuda"):
    #device = 'cpu'
    nt = t.shape[0]
    ncp = u.shape[0]-k-1
    
    N = torch.zeros(nt-2,ncp).to(device)
    
    for i in range(ncp):
        tab = BDP(t[1:-1],k,i,u)
        N[:,i] = tab[k,0,:]
    
    
    N2 = torch.zeros(nt,ncp).to(device)
    one = torch.tensor([1],dtype=torch.float64).view(1,-1)
    zeros = torch.zeros(ncp-1).view(1,-1)
    row1 = torch.cat((one,zeros),axis=1)
    row2 = torch.cat((zeros,one),axis=1)
    
    N2[0] = row1
    N2[-1] = row2
    N2[1:-1] = N
    
    return N2


def BDP(t,k,index,u,device="cuda"):
    
    #device = 'cpu'
    nT = t.shape[0]
    tab = torch.zeros(k+1,k+1,nT).to(device)
    
    #Level 0
    for i in range(k+1):
        mask = torch.zeros(nT).to(device)
        #B = torch.empty(nT)
        
        #indices = torch.where((t>=u[index+i]) & (t<u[index+i+1]))
        
        if index+i < len(u) - k -2:
            indices = torch.where((t>=u[index+i]) & (t<u[index+i+1]))
        else:
            indices = torch.where((t>=u[index+i]) & (t<=u[index+i+1])) 
        
        mask[indices] = 1
        y = (t+0.1)/(t+0.1)         #add 0.1 because t[0] = 0
        
        B = (y/y)*mask
        #B[0] = torch.tensor(0)
        tab[0,i,:] = B
    
    #level from 1 to k
    for l in torch.arange(1,k+1):
        for i in range(k+1-l):
            if u[index+i+l] == u[index+i]:
                c1 = 0.0
            else:
                #c1 = ((t-u[index+i])/(u[index+i+l]-u[index+i]))*tab[l-1,i,:].clone()
                c1 = ((t-u[index+i])/(u[index+i+l]-u[index+i]))*tab[l-1,i,:].clone()
            
            if u[index+i+l+1] == u[index+i+1]:
                c2 = 0.0
            else: 
                #c2 = ((u[index+i+l+1]-t)/(u[index+i+l+1]-u[index+i+1]))*tab[l-1,i+1,:].clone()
                c2 = ((u[index+i+l+1]-t)/(u[index+i+l+1]-u[index+i+1]))*tab[l-1,i+1,:].clone()
                
            #tab[l,i,:] = ((t-u[i])/(u[i+k]-u[i]))*tab[l-1,i,:] + ((u[i+k+1]-t)/(u[i+k+1]-u[i+1]))*tab[l-1,i+1,:]
            tab[l,i,:] = c1 + c2
    
    #return tab[k,0,:]
    return tab


def NDPWithBatch(t,k,u,device="cuda"):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    batchSize = t.shape[0]
    nt = t.shape[1]
    ncp = u.shape[0]-k-1
    
    N = torch.zeros(batchSize,nt,ncp).to(device)
    #N = torch.zeros(batchSize,nt-2,ncp).to(device)
    
    for i in range(ncp):
        tab = BDPWithBatch(t,k,i,u,device)
        #tab = BDPWithBatch(t[:,1:-1],k,i,u)
        N[:,:,i] = tab[:,k,0,:]
    
    
    return N

def NDPWithBatch2(t,k,u,device="cuda"):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    batchSize = t.shape[0]
    nt = t.shape[1]
    ncp = u.shape[0]-k-1
    
    #N = NDPWithBatch(t,k,u)
    
    N = torch.zeros(batchSize,nt-2,ncp).to(device)
    
    for i in range(ncp):
        tab = BDPWithBatch(t[:,1:-1],k,i,u,device)
        N[:,:,i] = tab[:,k,0,:]
    
    N2 = torch.zeros(batchSize,nt,ncp).to(device)
    one = torch.tensor([1],dtype=torch.float64).view(1,-1)
    zeros = torch.zeros(ncp-1).view(1,-1)
    row1 = torch.cat((one,zeros),axis=1)
    row2 = torch.cat((zeros,one),axis=1)
    
    N2[:,0] = row1
    N2[:,-1] = row2
    N2[:,1:-1] = N
    
    return N2

def NDPWithBatch3(t,k,u,device="cuda"):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    batchSize = t.shape[0]
    nt = t.shape[1]
    ncp = u.shape[1]-k-1
    
    N = torch.zeros(batchSize,nt-2,ncp).to(device)
    
    for i in range(ncp):
        tab = BDPWithBatch2(t[:,1:-1],k,i,u,device)
        N[:,:,i] = tab[:,k,0,:]
    
    
    return N

def BDPWithBatch(t,k,index,u,device="cuda"):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    batchSize = t.shape[0]
    nT = t.shape[1]
    tab = torch.zeros(batchSize,k+1,k+1,nT).to(device)
    
    #Level 0
    for i in range(k+1):
        mask = torch.zeros(batchSize,nT).to(device)
        #B = torch.empty(nT)
        
        #indices = torch.where((t>=u[index+i]) & (t<u[index+i+1]))
        
        if index+i < len(u) - k -2:
            indices = torch.where((t>=u[index+i]) & (t<u[index+i+1]))
        else:
            indices = torch.where((t>=u[index+i]) & (t<=u[index+i+1])) 
        
        mask[indices] = 1
        y = (t+0.1)/(t+0.1)         #add 0.1 because t[0] = 0
        
        B = (y/y)*mask
        #B[0] = torch.tensor(0)
        tab[:,0,i,:] = B
    
    #level from 1 to k
    for l in torch.arange(1,k+1):
        for i in range(k+1-l):
            if u[index+i+l] == u[index+i]:
                c1 = 0.0
            else:
                c1 = ((t-u[index+i])/(u[index+i+l]-u[index+i]))*tab[:,l-1,i,:].clone()
            
            if u[index+i+l+1] == u[index+i+1]:
                c2 = 0.0
            else: 
                c2 = ((u[index+i+l+1]-t)/(u[index+i+l+1]-u[index+i+1]))*tab[:,l-1,i+1,:].clone()
                
            #tab[l,i,:] = ((t-u[i])/(u[i+k]-u[i]))*tab[l-1,i,:] + ((u[i+k+1]-t)/(u[i+k+1]-u[i+1]))*tab[l-1,i+1,:]
            tab[:,l,i,:] = c1 + c2
    
    #return tab[k,0,:]
    return tab

def BDPWithBatch2(t,k,index,u,device="cuda"):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    batchSize = t.shape[0]
    nT = t.shape[1]
    tab = torch.zeros(batchSize,k+1,k+1,nT).to(device)
    
    #Level 0
    for i in range(k+1):
        mask = torch.zeros(batchSize,nT).to(device)
        #B = torch.empty(nT)
        
        #indices = torch.where((t>=u[index+i]) & (t<u[index+i+1]))
        
        if index+i < u.shape[1] - k -2:
            indices = torch.where((t>=u[:,index+i].view(-1,1)) & (t<u[:,index+i+1].view(-1,1)))
        else:
            indices = torch.where((t>=u[:,index+i].view(-1,1)) & (t<=u[:,index+i+1].view(-1,1))) 
        
        mask[indices] = 1
        y = (t+0.1)/(t+0.1)         #add 0.1 because t[0] = 0
        
        B = (y/y)*mask
        #B[0] = torch.tensor(0)
        tab[:,0,i,:] = B
    
    #level from 1 to k
    for l in torch.arange(1,k+1):
        for i in range(k+1-l):
            
            cond1 = u[:,index+i+l] != u[:,index+i]
            cond2 = u[:,index+i+l+1] != u[:,index+i+1]
            
            indices1 = torch.where((cond1 == True) & (cond2 == False))
            indices2 = torch.where((cond1 == False) & (cond2 == True))
            indices3 = torch.where((cond1 == True) & (cond2 == True))
            indices4 = torch.where((cond1 == False) & (cond2 == False))
            
            
            num1 = (t-u[:,index+i].view(-1,1))
            den1 = (u[:,index+i+l].view(-1,1)-u[:,index+i].view(-1,1))
            
            num2 = (u[:,index+i+l+1].view(-1,1)-t) 
            den2 = (u[:,index+i+l+1].view(-1,1)-u[:,index+i+1].view(-1,1))
            
            c1 = (num1/den1)*tab[:,l-1,i,:].clone()
            c2 = (num2/den2)*tab[:,l-1,i+1,:].clone()
            
            #tab[l,i,:] = ((t-u[i])/(u[i+k]-u[i]))*tab[l-1,i,:] + ((u[i+k+1]-t)/(u[i+k+1]-u[i+1]))*tab[l-1,i+1,:]
            
            ind1 = indices1[0]
            ind2 = indices2[0]
            ind3 = indices3[0]
            ind4 = indices4[0]
            
            if len(indices1) > 0:
                tab[ind1,l,i,:] = c1[ind1]
            
            if len(indices2) > 0:
                tab[ind2,l,i,:] = c2[ind2]
            
            if len(indices3) > 0:
                tab[ind3,l,i,:] = c1[ind3] + c2[ind3]
                
            if len(indices4) > 0:    
                tab[ind4,l,i,:] = torch.zeros(len(indices4))
    
    #return tab[k,0,:]
    return tab

