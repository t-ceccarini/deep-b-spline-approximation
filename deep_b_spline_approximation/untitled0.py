# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 00:45:18 2022

@author: Tommaso
"""

from deep_b_spline_approximation import BSplineApproximator
import torch
from loadfromtxt import loadFlatDataset

PATHEVAL = r"parnet_datasets\evalset1.txt"

flatevalset,evalset = loadFlatDataset(PATHEVAL,2)
flatevaltorch = torch.tensor(flatevalset)

ncurveval = evalset.shape[0] // 2
npoints = evalset.shape[1] 

eval1 = torch.tensor(evalset)
eval2 = eval1.reshape(ncurveval,2,npoints)
curves = eval2.permute(0,2,1)
curves = curves[:10].to("cpu")

app = BSplineApproximator(device='cpu')

print(f"curves device {curves.device}")
print(f"app device {app.ppn.device}")

splines,list_of_knots,HD_log,MSE_log = app.approximate(curves,1)