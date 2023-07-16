# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:38:15 2022

@author: Tommaso
"""
import sys
import urllib.request
import os

def download_file(model_url, dest):
    
    print(f"Downloading {model_url}")
    
    sys.stdout.flush()
    urllib.request.urlretrieve(model_url,dest)
    
    print(f"Downloaded {model_url}")

def download_models(output_dir):
    print("Downloading models")
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/ppn_mlp1.pt?raw=true',
                  os.path.join(output_dir,'ppn_mlp1.pt'))
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/kpn_mlp4.pt?raw=true',
                  os.path.join(output_dir,'kpn_mlp4.pt'))
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/ppn_cnn1.pt?raw=true',
                  os.path.join(output_dir,'ppn_cnn1.pt'))
    
    print("Downloaded models")
    
def download_evalsets(output_dir):
    print("Downloading evalsets")
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/evalsets/evalset1.txt?raw=true',
                  os.path.join(output_dir,'evalset1.txt'))
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/evalsets/evalset2.txt?raw=true',
                  os.path.join(output_dir,'evalset2.txt'))
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/evalsets/evalset3.txt?raw=true',
                  os.path.join(output_dir,'evalset3.txt'))
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/evalsets/evalset4.txt?raw=true',
                  os.path.join(output_dir,'evalset4.txt'))
    
    print("Downloaded evalsets")

def main():
    
    if sys.argv[1] == 'download-models':
        os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)),'models'))
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')
        download_models(output_dir)
        
        return
    
    elif sys.argv[1] == 'download-evalsets':
        os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)),'evalsets'))
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'evalsets')
        download_evalsets(output_dir)
        
        return

if __name__ == "__main__":
    main()
