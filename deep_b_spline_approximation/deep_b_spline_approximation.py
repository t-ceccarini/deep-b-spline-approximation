# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:45:12 2022

@author: Tommaso
"""
import torch
import requests
from deep_b_spline_approximation.preprocessing import computeSegmentation,computeSampling,computeSegmentsNormalization,computeSegmentsParametrization,computeRefinement2
from deep_b_spline_approximation.ppn import PointParametrizationNetwork,PointParametrizationNetworkCNN2
from deep_b_spline_approximation.kpn import KnotPlacementNetwork

class BSplineApproximator:
    
    def __init__(self, ppn_type="mlp", device='cpu'):
        
        if(device == 'cuda'):
            if(torch.cuda.is_available()):
                self.device = torch.device('cuda')
            else:
                raise Exception("GPU not available.")
        elif(device == 'cpu'):
            self.device = 'cpu'
        
        #download KPN-MLP model
        kpn_url = 'https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/kpn_mlp4.pt'
        r = requests.get(kpn_url, allow_redirects=True)
        open('kpn_mlp4.pt','wb').write(r.content)
        
        self.path_load_kpn = r"kpn_mlp4.pt"
        
        self.p = 2
        
        #Load point parametrization network
        if(ppn_type == "mlp"):
            #download PPN-MLP model
            ppn_url = 'https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/ppn_mlp1.pt'
            r = requests.get(ppn_url, allow_redirects=True)
            open('ppn_mlp1.pt','wb').write(r.content)
            
            self.path_load_ppn = r"ppn_mlp1.pt"
            dim, hiddenSize = 200, 1000
            self.ppn = PointParametrizationNetwork(dim,hiddenSize,self.p,device=self.device)
        elif(ppn_type == "cnn"):
            self.path_load_ppn = r"models\ppn_cnn1.pt"
            self.ppn = PointParametrizationNetworkCNN2(device=self.device)
            
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

        points = points.to(self.device)
        
        splines = torch.zeros(points.shape[0],points.shape[1],points.shape[2]).to(self.device)
        list_of_knots = torch.zeros(points.shape[0],2*(k+1)+n_knots).to(self.device)
        HD_log = torch.zeros(points.shape[0],n_knots+1).to(self.device)
        MSE_log = torch.zeros(points.shape[0],n_knots+1).to(self.device)
        
        for i,seq_of_points in enumerate(points,0):
            
            segs = segments[i]

            l = points.shape[1]

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
                                                                                                       self.device)

            spline,knots,cps,hd,mse = computeRefinement2(seq_of_points,
                                                                   seq_of_points_seg_norm,
                                                                   param,
                                                                   param_seg_norm,
                                                                   knots,
                                                                   ranges,
                                                                   k,
                                                                   n_knots,
                                                                   self.kpn,
                                                                   self.device)
            
            splines[i] = spline
            list_of_knots[i] = knots
            HD_log[i] = hd
            MSE_log[i] = mse
            
            torch.cuda.empty_cache()
            
            print("sequence of points {}, Hausdorff distance: {:0.4f}, Mean Squared Error {:0.4f}".format(i,hd[-1],mse[-1]))
            
        return splines,list_of_knots,HD_log,MSE_log