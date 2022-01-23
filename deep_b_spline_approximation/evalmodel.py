# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:40:20 2020

@author: Tommaso
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ppn import PointParametrizationNetwork,PointParametrizationNetworkLSTM,PointParametrizationNetworkCNN2,EncoderCNN,DecoderCNN,Seq2SeqCNN
from loadfromtxt import loadFlatDataset
from parametrization import computeCentripetalParametrization,computeChordLengthParametrization,computeUniformParametrization
from BSpline import computeBSplineApproximation

torch.set_default_dtype(torch.float64)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

#PATHLOAD = r"models\ppn_mlp1.pt"
PATHLOAD = r"models\ppn_cnn1.pt"
#PATHLOAD = r"models\ppn_rnn1.pt"
#PATHLOAD = r"models\ppn_seq2seq_cnn1.pt"
#PATHLOAD = r"models\ppn_seq2seq_rnn1.pt"

#ppn = PointParametrizationNetwork(200,1000,2,device="cpu")
ppn = PointParametrizationNetworkCNN2(device="cpu") 
#ppn = PointParametrizationNetworkLSTM(device="cpu")

"""
encoder = EncoderCNN(2,64,128,3,7,0.25,device).to(device)
decoder = DecoderCNN(2,1,64,128,3,7,0.25,0,device).to(device)
ppn = Seq2Seq(encoder,decoder).to(device)
"""
"""
INPUT_DIM = 2
OUTPUT_DIM = 1
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(INPUT_DIM, OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
ppn = Seq2Seq(enc, dec, device)
"""

#ppn.to(device)

checkpoint = torch.load(PATHLOAD)
ppn.load_state_dict(checkpoint['model_state_dict'])

ppn.eval()

PATHEVAL = r"parnet_datasets\evalset2.txt"
flatevalset,evalset = loadFlatDataset(PATHEVAL,2)
flatevaltorch = torch.tensor(flatevalset)

eval1 = torch.tensor(evalset)
eval2 = eval1.reshape(500,2,100)
eval3 = eval2.permute(0,2,1)

#indmaxhdcentr = torch.tensor([327,  79, 358, 255], device='cuda:0')
#indmaxhdcentrval2 = torch.tensor([ 79, 327, 153, 443], device='cuda:0')
#indmaxhdchordval2 = torch.tensor([ 79, 327, 153, 358], device='cuda:0')
#indmaxhduniformval2 = torch.tensor([ 79, 327, 443, 153], device='cuda:0')
#eval3 = eval3[indmaxhdcentr]
#flateval3 = flatevaltorch[indmaxhdcentr]

flateval3 = flatevaltorch.clone()

#splines,t = ppn(eval3,eval3)
splines,t,curves = ppn(eval3)
#splines,t,curves = ppn(flateval3)

#curves = eval3.cuda()
curves = eval3.clone()

splines = computeBSplineApproximation(curves,t,"cpu")

tcentr = computeCentripetalParametrization(curves)
tchord = computeChordLengthParametrization(curves)
tuni = computeUniformParametrization(curves)

splinescentr = computeBSplineApproximation(curves,tcentr.cpu(),"cpu")
splineschord = computeBSplineApproximation(curves,tchord.cpu(),"cpu")
splinesuni = computeBSplineApproximation(curves,tuni.cpu(),"cpu")

#avghdcentr,hdcentr,i3,i4 = DHDLoss(curves,splinescentr)

#maxhd,indmaxhd = torch.topk(hd,4)
#maxhdcentr,indmaxhdcentr = torch.topk(hdcentr,4)

#indmaxhdcentr = torch.tensor([ 79, 327, 443, 153], device='cuda:0')
#indmaxhdcentr = torch.tensor([ 32, 145, 313, 473], device='cuda:0')

#indmaxhdcentr = torch.tensor([12,48,296,442], device='cuda:0')          

indmaxhdcentr = torch.tensor([199,99,9,299], device='cuda:0')

curvesnp = curves[indmaxhdcentr].cpu().numpy()
splinesnp = splines[indmaxhdcentr].cpu().detach().numpy()

splinescentrnp = splinescentr[indmaxhdcentr].cpu().detach().numpy()
splineschordnp = splineschord[indmaxhdcentr].cpu().detach().numpy()
splinesunidnp = splinesuni[indmaxhdcentr].cpu().detach().numpy()


"""
curvesnp = curves.cpu().numpy()
splinesnp = splines.cpu().detach().numpy()
splinescentrnp = splinescentr.cpu().detach().numpy()
"""

x1,y1 = curvesnp[:,:,0],curvesnp[:,:,-1]
x2,y2 = splinesnp[:,:,0],splinesnp[:,:,-1]
x3,y3 = splinescentrnp[:,:,0],splinescentrnp[:,:,-1]
x4,y4 = splineschordnp[:,:,0],splineschordnp[:,:,-1]
x5,y5 = splinesunidnp[:,:,0],splinesunidnp[:,:,-1]


"""
plt.plot(x1[3],y1[3],'.',c='k')
plt.plot(x2[0],y2[0],c='b',linewidth=2.0)
plt.plot(x3[0],y3[0],c='r',linewidth=2.0)
plt.plot(x4[0],y4[0],c='g',linewidth=2.0)
plt.plot(x5[0],y5[0],c='y',linewidth=2.0)
plt.axis('off')

path="bezier-eval2-seq4.pdf"
plt.tight_layout()
plt.savefig(path)
plt.show()

"""

fig,axes = plt.subplots(2,2,figsize=(10,10))  


#plt.plot(x1[0],y1[0],'.',c='k')
#plt.plot(x2[0],y2[0],c='b',linewidth=2.0)
#plt.plot(x3[0],y3[0],c='r',linewidth=2.0)
#plt.plot(x4[0],y4[0],c='g',linewidth=2.0)
#plt.plot(x5[0],y5[0],c='y',linewidth=2.0)
#plt.axis('off')


axes[0,0].plot(x1[0],y1[0],'.',c='k')
axes[0,0].plot(x2[0],y2[0],c='b',linewidth=2.0)
axes[0,0].plot(x3[0],y3[0],c='r',linewidth=2.0)
axes[0,0].plot(x4[0],y4[0],c='g',linewidth=2.0)
axes[0,0].plot(x5[0],y5[0],c='y',linewidth=2.0)
axes[0,0].axis('off')

#plt.axis('off')

#axes[0,0].plot(x1[4],y1[4],'.',c='k')
#axes[0,0].plot(x2[4],y2[4],c='b',linewidth=2.0)
#axes[0,0].plot(x3[4],y3[4],c='r',linewidth=2.0)

axes[0,1].plot(x1[1],y1[1],'.',c='k')
axes[0,1].plot(x2[1],y2[1],c='b',linewidth=2.0)
axes[0,1].plot(x3[1],y3[1],c='r',linewidth=2.0)
axes[0,1].plot(x4[1],y4[1],c='g',linewidth=2.0)
axes[0,1].plot(x5[1],y5[1],c='y',linewidth=2.0)
axes[0,1].axis('off')

#plt.axis('off')

axes[1,0].plot(x1[2],y1[2],'.',c='k')
axes[1,0].plot(x2[2],y2[2],c='b',linewidth=2.0)
axes[1,0].plot(x3[2],y3[2],c='r',linewidth=2.0)
axes[1,0].plot(x4[2],y4[2],c='g',linewidth=2.0)
axes[1,0].plot(x5[2],y5[2],c='y',linewidth=2.0)
axes[1,0].axis('off')

#plt.axis('off')

#axes[1,0].plot(x1[5],y1[5],'.',c='k')
#axes[1,0].plot(x2[5],y2[5],c='b',linewidth=2.0)
#axes[1,0].plot(x3[5],y3[5],c='r',linewidth=2.0)

axes[1,1].plot(x1[3],y1[3],'.',c='k')
axes[1,1].plot(x2[3],y2[3],c='b',linewidth=2.0)
axes[1,1].plot(x3[3],y3[3],c='r',linewidth=2.0)
axes[1,1].plot(x4[3],y4[3],c='g',linewidth=2.0)
axes[1,1].plot(x5[3],y5[3],c='y',linewidth=2.0)
axes[1,1].axis('off')

#plt.axis('off')
#path="plots/sequenze-example1.pdf"
"""
plt.tight_layout()
plt.savefig(path)

plt.show()
"""
