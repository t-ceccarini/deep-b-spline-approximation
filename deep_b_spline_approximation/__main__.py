# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:38:15 2022

@author: Tommaso
"""
import sys
import requests

def download_file(model_url, dest):
    
    print(f"Downloading {model_url}")
    
    r = requests.get(model_url, allow_redirects=True)
    open(dest,'wb').write(r.content)
    
    print(f"Downloaded {model_url}")

def download_models():
    print("Downloading models")
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/ppn_mlp1.pt?raw=true',
                  'ppn_mlp1.pt')
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/kpn_mlp4.pt?raw=true',
                  'kpn_mlp4.pt')
    
    download_file('https://github.com/t-ceccarini/deep-b-spline-approximation/blob/master/models/ppn_cnn1.pt?raw=true',
                  'ppn_cnn1.pt')
    
    print("Downloaded models")
    

def main():
    
    if sys.argv[1] == 'downloads-models':
        download_models()
        
        return

if __name__ == "main":
    main()
