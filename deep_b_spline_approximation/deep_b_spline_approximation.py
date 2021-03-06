# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:45:12 2022

@author: Tommaso
"""
import torch
from prettytable import PrettyTable
import os
from .preprocessing import computeSegmentation,computeSampling,computeSegmentsNormalization,computeSegmentsParametrization,computeRefinement2
from .ppn import PointParametrizationNetwork,PointParametrizationNetworkCNN2
from .kpn import KnotPlacementNetwork

class BSplineApproximator:
    
    def __init__(self, ppn_type="mlp", device='cpu'):
        
        if(device == 'cuda'):
            if(torch.cuda.is_available()):
                self.device = torch.device('cuda')
            else:
                raise Exception("GPU not available.")
        elif(device == 'cpu'):
            self.device = 'cpu'
        
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')
        
        self.path_load_kpn = os.path.join(output_dir,'kpn_mlp4.pt')
        
        self.p = 2
        
        #Load point parametrization network
        if(ppn_type == "mlp"):
            self.path_load_ppn = os.path.join(output_dir,'ppn_mlp1.pt')
            dim, hiddenSize = 200, 1000
            self.ppn = PointParametrizationNetwork(dim,hiddenSize,self.p,device=self.device)
        elif(ppn_type == "cnn"):
            self.path_load_ppn = os.path.join(output_dir,'ppn_cnn1.pt')
            self.ppn = PointParametrizationNetworkCNN2(device=self.device)
        
        self.ppn_type = ppn_type
        checkpoint1 = torch.load(self.path_load_ppn)
        self.ppn.load_state_dict(checkpoint1['model_state_dict'])
        self.ppn = self.ppn.eval()
        
        #Load knot placement network
        self.kpn = KnotPlacementNetwork(inputSize=300,hiddenSize=500,p=self.p,k=3,device=self.device)
        self.kpn.to(self.device)

        checkpoint2 = torch.load(self.path_load_kpn)
        self.kpn.load_state_dict(checkpoint2['model_state_dict'])
        self.kpn = self.kpn.eval()
        
        #We use the 98-percentile of the Parnet training set distribution of complexity for the segmentation of the sequences
        self.thresholdPerc = torch.tensor(11.5373).to(self.device)
        
        
    def approximate(self, points, n_knots, k=3):
        
        self.thresholdPerc,segments,itosplit = computeSegmentation(points,self.thresholdPerc)
        n_max_knots = max([len(s) for s in segments])

        points = points.to(self.device)
        
        l = 100
        
        length = n_max_knots if n_max_knots >= n_knots else n_knots
        
        splines = torch.zeros(points.shape[0],points.shape[1],points.shape[2]).to(self.device)
        list_of_knots = torch.zeros(points.shape[0],2*(k+1)+length).to(self.device)
        HD_log = torch.zeros(points.shape[0],length+1).to(self.device)
        MSE_log = torch.zeros(points.shape[0],length+1).to(self.device)
        
        t = PrettyTable(['Sequence of points n.','Degree of the spline function','Hausdorff distance','Mean Squared Error'])
        t.align['Sequence of points n.'] = 'l'
        t.align['Degree of the spline function'] = 'l'
        t.align['Hausdorff distance'] = 'l'
        t.align['Mean Squared Error'] = 'r'
        t.hrules = 1
        print(t)
        
        for i,seq_of_points in enumerate(points,0):
            
            segs = segments[i]

            seq_of_points_seg, ranges, indices = computeSampling(seq_of_points,
                                                                 segs,
                                                                 l,
                                                                 self.device)
            
            seq_of_points_seg_norm = computeSegmentsNormalization(seq_of_points_seg,
                                                                  seq_of_points,
                                                                  ranges,
                                                                  indices,
                                                                  l)
            
            param,param_seg_norm_rescaled,param_seg_norm,knots = computeSegmentsParametrization(seq_of_points,
                                                                                                       seq_of_points_seg,
                                                                                                       seq_of_points_seg_norm,
                                                                                                       ranges,
                                                                                                       indices,
                                                                                                       l,
                                                                                                       self.ppn,
                                                                                                       self.ppn_type,
                                                                                                       self.device)

            spline,knots,cps,hd,mse = computeRefinement2(seq_of_points,
                                                                   seq_of_points_seg_norm,
                                                                   param,
                                                                   param_seg_norm,
                                                                   knots,
                                                                   ranges,
                                                                   k,
                                                                   length,
                                                                   n_knots,
                                                                   self.kpn,
                                                                   self.device)
            
            splines[i] = spline
            list_of_knots[i] = knots
            HD_log[i] = hd
            MSE_log[i] = mse
            
            torch.cuda.empty_cache()
            
            t.add_row([i,k,round(hd[-1].item(),3),round(mse[-1].item(),3)])
            print ("\n".join(t.get_string().splitlines()[-2:]))
            
        return splines,list_of_knots,HD_log,MSE_log