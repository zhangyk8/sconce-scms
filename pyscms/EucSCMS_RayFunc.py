#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: June 19, 2022

Description: This script contains code for the parallel implementations with 
Ray (https://www.ray.io/) of the standard (or Euclidean) kernel density estimator 
with the Gaussian kernel, its gradient estimator, mean shift (MS), and subspace 
constrained mean shift (SCMS) algorithms.
"""

import numpy as np
from numpy import linalg as LA
import ray

#==========================================================================================#

@ray.remote
def KDE_Ray(x, data, h=None, wt=None):
    '''
    The d-dim Euclidean KDE with Gaussian kernel.
    
    Parameters:
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
    
    Return:
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


@ray.remote
def MS_Ray(mesh_0, data, h=None, eps=1e-7, max_iter=1000, wt=None):
    '''
    Mean Shift Algorithm with the Gaussian kernel.
    
    Parameters:
        mesh_0: a (m,d)-array
            The coordinates of m initial points in the d-dim Euclidean space.
    
        data: a (n,d)-array
            The coordinates of n data sample points in the d-dim Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
        
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
    
    Return:
        MS_new: (m,d)-array
            The collection of converged points yielded by the Euclidean SCMS 
            algorithm.
    '''
    
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        # (Only works for Gaussian kernel)
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
        print("The current bandwidth is "+ str(h) + ".\n")
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    if wt is None:
        wt = np.ones((n,))
    else:
        wt = n*wt
    MS_new = np.copy(mesh_0)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            # print('The MS algorithm converges in ' + str(t-1) + 'steps!')
            break
        MS_old = np.copy(MS_new)
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pt = MS_old[i,:]
                ker_w = wt*np.exp(-np.sum(((x_pt-data)/h)**2, axis=1)/2)
                if np.sum(ker_w) == 0:
                    # Set those points with zero density values to NaN
                    nan_arr = np.zeros_like(x_pt)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    # Mean shift update
                    x_new = np.sum(data*ker_w.reshape(n,1), axis=0) / np.sum(ker_w)
                    if LA.norm(x_pt - x_new) < eps:
                        conv_sign[i] = 1
                MS_new[i,:] = x_new
            else:
                MS_new[i,:] = MS_old[i,:]
    
    '''
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    '''
    MS_new = MS_new[conv_sign == 1]
    # Drop NaN entries before returning the mode candidates
    return MS_new[~np.isnan(MS_new[:,0])]


@ray.remote
def SCMSLog_Ray(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, wt=None,
                stop_cri='proj_grad'):
    '''
    Subspace Constrained Mean Shift algorithm with the log density and Gaussian 
    kernel.
    
    Parameters:
        mesh_0: a (m,D)-array
            The coordinates of m initial points in the D-dim Euclidean space.
    
        data: a (n,D)-array
            The coordinates of n data sample points in the D-dim Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
       
        stop_cri: string ('proj_grad'/'pts_diff')
            The indicator of which stopping criteria that will be used to 
            terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors 
            between two consecutive iteration points need to be smaller than 
            'eps' for terminating the algorithm. When stop_cri='proj_grad' or 
            others, the projected/principal gradient of the current point need to be 
            smaller than 'eps' for terminating the algorithm.)
            (Default: stop_cri='proj_grad'.)
    
    Return:
        SCMS_new: (m,D)-array
            The collection of converged points yielded by the Euclidean SCMS 
            algorithm.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Dimension of data points
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        # (Only works for Gaussian kernel)
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data, axis=0))
        print("The current bandwidth is "+ str(h) + ".\n")
    
    SCMS_new = np.copy(mesh_0)
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    if wt is None:
        wt = np.ones((n,1))
    else:
        wt = wt.reshape(n,1)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            # print('The SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        SCMS_old = np.copy(SCMS_new)
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_old[i,:]
                ## Compute the gradient of the log density
                Grad = np.sum(-wt*(x_pts - data) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) /(h**2)
                den_prop = np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2))
                if den_prop == 0:
                    # Set those points with zero density values to NaN
                    nan_arr = np.zeros_like(x_pts)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    Log_grad = Grad / den_prop
                    ## Compute the Hessian matrix
                    Log_Hess = np.dot((x_pts-data).T, wt*(x_pts-data) \
                                  * np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1)) \
                               /(h**4 * np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2))) \
                    - np.diag(np.ones(len(x_pts,)) / (h**2)) - np.dot(Log_grad.reshape(D,1), Log_grad.reshape(1,D))
                    Log_Hess = np.dot((x_pts-data).T, wt*(x_pts-data) \
                                  * np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1)) /den_prop \
                    - np.diag(np.ones(len(x_pts,)) * (h**2)) - np.dot((h**2)*Log_grad.reshape(D,1), (h**2)*Log_grad.reshape(1,D))
                    ## Spectral decomposition
                    w, v = LA.eig(Log_Hess)
                    ## Obtain the eigenpairs
                    V_d = v[:, np.argsort(w)[:(len(x_pts)-d)]]
                    ## Mean Shift vector
                    ms_v = np.sum(wt*data*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) \
                           / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) - x_pts
                    ## Subspace constrained gradient and mean shift vector
                    SCMS_grad = np.dot(V_d, np.dot(V_d.T, Log_grad))
                    SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                    ## SCMS update
                    x_new = SCMS_v + x_pts
                    ## Stopping criteria
                    if stop_cri == 'pts_diff':
                        if LA.norm(SCMS_v) < eps:
                            conv_sign[i] = 1
                    else: 
                        if LA.norm(SCMS_grad) < eps:
                            conv_sign[i] = 1
                    if sum(x_new.imag) > 0:
                        # Remove those points with nonzero imaginary parts
                        nan_arr = np.zeros(x_pts.shape[0], )
                        nan_arr[:] = np.nan
                        conv_sign[i] = 1
                        x_new = nan_arr
                SCMS_new[i,:] = x_new.real
            else:
                SCMS_new[i,:] = SCMS_old[i,:]
        # print(t)
    
    '''
    if t >= max_iter-1:
        print('The SCMS algorithm reaches the maximum number of iterations,'\
              +str(max_iter)+', and has not yet converged.')
    '''
    SCMS_new = SCMS_new[conv_sign == 1]
    # Drop NaN entries before returning the ridge candidates
    return SCMS_new[~np.isnan(SCMS_new[:,0])]
