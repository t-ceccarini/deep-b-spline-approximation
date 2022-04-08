# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 23:29:22 2022

@author: Tommaso
"""
from setuptools import setup

VERSION = '0.2.3'
DESCRIPTION = 'A python package for bspline curve approximation using deep learning'

# Setting up
setup(
    name='deep-b-spline-approximation',
    packages=['deep_b_spline_approximation'],
    version=VERSION,
    author="Tommaso Ceccarini",
    author_email="<tceccarini93@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/t-ceccarini/deep-b-spline-approximation',
    download_url='https://github.com/t-ceccarini/deep-b-spline-approximation/archive/refs/tags/v_0.2.4.tar.gz',
    install_requires=['torch','prettytable','numpy','scipy','matplotlib'],
    keywords=['python', 'deep learning', 'mlp', 'cnn', 'cagd', 'bspline', 'bezier'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
