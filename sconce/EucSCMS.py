#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Yikun Zhang
'''

# Author: Yikun Zhang
# Last Editing: June 26, 2022

# Description: This script contains code for the standard (or Euclidean) 
# kernel density estimator with the Gaussian kernel, its gradient estimator, 
# mean shift (MS), and subspace constrained mean shift (SCMS) algorithms.

import numpy as np
from numpy import linalg as LA

#==============================================================================#

def KDE(x, data, h=None, wt=None):
    '''
    The d-dim Euclidean KDE with the Gaussian kernel.
    
    Parameters:
    ----------
        x: (m,d)-array
            The coordinates of m query points in the d-dim Euclidean space.
    
        data: (n,d)-array
            The coordinates of n random sample points in the d-dimensional 
            Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
    
    Returns:
    ----------
        f_hat: (m,)-array
            The corresponding kernel density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Dimension of the data
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        # (Only works for Gaussian kernel)
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
        print("The current bandwidth is "+ str(h) + ".\n")
    
    f_hat = np.zeros((x.shape[0], ))
    if wt is None:
        wt = np.ones((n,))/n
    for i in range(x.shape[0]):
        f_hat[i] = np.sum(wt*np.exp(np.sum(-((x[i,:] - data)/h)**2, axis=1)/2))/ \
                   ((2*np.pi)**(d/2)*np.prod(h))
    return f_hat
