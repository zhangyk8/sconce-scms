#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: June 19, 2022

Description: This script contains code for the parallel implementations with 
Ray (https://www.ray.io/) of the directional-linear kernel density 
estimator with the von Mises and Gaussian kernels, directional-linear mean 
shift (DirLinMS), and directional-linear subspace constrained mean shift 
(DirLinSCMS) algorithms.
"""

import numpy as np
from numpy import linalg as LA
import scipy.special as sp
import ray

#==========================================================================================#

@ray.remote
def DirLinKDE_Ray(x, data, h=None, b=None, q=2, D=1):
    '''
    Directional-linear KDE with the von Mises and Gaussian kernels
    
    Parameters:
        x: (m, q+1+D)-array
            Eulidean coordinates of m directional-linear query points, where 
            (q+1) is the Euclidean dimension of the directional component (first 
            (q+1) columns) and D is the dimension of the linear component (last 
            D columns).
    
        data: (n, q+1+D)-array
            Euclidean coordinates of n directional-linear random sample points, 
            where (q+1) is the Euclidean dimension of the directional component 
            (first (q+1) columns) and D is the dimension of the linear component
            (last D columns).
       
        h: float
            Bandwidth parameter for the directional component. (Default: h=None. 
            Then a rule of thumb for directional KDEs with the von Mises kernel 
            in Garcia-Portugues (2013) is applied.)
            
        b: float
            Bandwidth parameter for the linear component. (Default: h=None. 
            Then the Silverman's rule of thumb is applied. See Chen et al.(2016) 
            for details.)
            
        q: int
            Intrinsic data dimension of directional components.
            
        D: int
            Data dimension of linear components.
    
    Return:
        f_hat: (m,)-array
            The corresponding directinal-linear density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    assert q+1+D == D_t, "The dimension of the input data, "+str(D_t)+", should "\
    "equal to the sum of directional dimension ("+str(q+1)+") in its ambient "\
    "space and linear dimension ("+str(D)+")."
    
    data_Dir = data[:,:(q+1)]
    x_Dir = x[:,:(q+1)]
    data_Lin = data[:,(q+1):(q+1+D)]
    x_Lin = x[:,(q+1):(q+1+D)]

    # Rule of thumb for directional component
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
        if q == 2:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
                 (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
                                  (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
        print("The current bandwidth for directional component is " + str(h) + ".\n")
    
    # Rule of thumb for linear component
    if b is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter
        b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
        print("The current bandwidth for linear component is "+ str(b) + ".\n")
    
    # Compute the kernel weights contributed by directional components
    if q == 2:
        f_hat_Dir = np.exp((np.dot(x_Dir, data_Dir.T)-1)/(h**2))/(2*np.pi\
                        *(1-np.exp(-2/h**2))*h**2)
    else:
        f_hat_Dir = np.exp(np.dot(x_Dir, data_Dir.T)/(h**2))/((2*np.pi)**((q+1)/2)*\
                           sp.iv((q-1)/2, 1/(h**2))*h**(q-1))
    
    # Compute the kernel weights contributed by linear components
    f_hat_Lin = np.zeros((x.shape[0], n))
    for i in range(x.shape[0]):
        f_hat_Lin[i,:] = np.exp(np.sum(-((x_Lin[i,:] - data_Lin)/b)**2, axis=1)/2)/ \
                        ((2*np.pi)**(D/2)*np.prod(b))
    
    f_hat = np.mean(f_hat_Dir * f_hat_Lin, axis=1)
    return f_hat


@ray.remote
def DirLinMS_Ray(mesh_0, data, h=None, b=None, q=2, D=1, eps=1e-7, max_iter=1000):
    '''
    Directional-linear Mean Shift Algorithm with the von Mises and Gaussian kernels
    (Simultaneous version).
    
    Parameters:
        mesh_0: (m, q+1+D)-array
            Eulidean coordinates of m directional-linear query points, where 
            (q+1) is the Euclidean dimension of the directional component (first 
            (q+1) columns) and D is the dimension of the linear component (last 
            D columns).
    
        data: (n, q+1+D)-array
            Euclidean coordinates of n directional-linear random sample points, 
            where (q+1) is the Euclidean dimension of the directional component 
            (first (q+1) columns) and D is the dimension of the linear component
            (last D columns).
       
        h: float
            Bandwidth parameter for the directional component. (Default: h=None. 
            Then a rule of thumb for directional KDEs with the von Mises kernel 
            in Garcia-Portugues (2013) is applied.)
            
        b: float
            Bandwidth parameter for the linear component. (Default: h=None. 
            Then the Silverman's rule of thumb is applied. See Chen et al.(2016) 
            for details.)
            
        q: int
            Intrinsic data dimension of directional components. (Default: q=2.)
            
        D: int
            Data dimension of linear components. (Default: D=1.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000.)
    
    Return:
        MS_new: (m,q+1+D)-array
            The collection of converged points yielded by the DirLinMS algorithm.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    assert q+1+D == D_t, "The dimension of the input data, "+str(D_t)+", should "\
    "equal to the sum of directional dimension ("+str(q+1)+") in its ambient "\
    "space and linear dimension ("+str(D)+")."
    
    data_Dir = data[:,:(q+1)]
    data_Lin = data[:,(q+1):(q+1+D)]

    # Rule of thumb for directional component
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
        # An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
        if q == 2:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
                 (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
                                  (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
        print("The current bandwidth for directional component is " + str(h) + ".\n")
    
    # Rule of thumb for linear component
    if b is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
        print("The current bandwidth for linear component is "+ str(b) + ".\n")
    
    MS_new = np.copy(mesh_0)
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            # print('The directional-linear MS algorithm converges in ' + str(t-1) + 'steps!')
            break
        MS_old = np.copy(MS_new)
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_Dir = MS_old[i,:(q+1)]
                x_Lin = MS_old[i,(q+1):(q+1+D)]
                # Kernel weights
                ker_w_Dir = np.exp((np.dot(data_Dir, x_Dir) - 1)/ h**2)
                ker_w_Lin = np.exp(-np.sum(((x_Lin - data_Lin)/b)**2, axis=1)/2)
                # Mean shift updates for directional components
                x_Dir_new = np.sum(data_Dir * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1), 
                                   axis=0)
                x_Dir_new = x_Dir_new / LA.norm(x_Dir_new)
                MS_new[i,:(q+1)] = x_Dir_new
                # Mean shift updates for linear components
                x_Lin_new = np.sum(data_Lin * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1),
                                   axis=0) / np.sum(ker_w_Dir * ker_w_Lin)
                MS_new[i,(q+1):(q+1+D)] = x_Lin_new
                if LA.norm(MS_new - MS_old) <= eps:
                    conv_sign[i] = 1
            else:
                MS_new[i,:] = MS_old[i,:]
                
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    MS_new = MS_new[conv_sign == 1]
    # Drop NaN entries before returning the mode candidates
    return MS_new[~np.isnan(MS_new[:,0])]

                
@ray.remote             
def DirLinMS_CA_Ray(mesh_0, data, h=None, b=None, q=2, D=1, eps=1e-7, max_iter=1000):
    '''
    Directional-Linear Mean Shift Algorithm with the von Mises and Gaussian kernels
    (Componentwise version).
    
    Parameters:
        mesh_0: (m, q+1+D)-array
            Eulidean coordinates of m directional-linear query points, where 
            (q+1) is the Euclidean dimension of the directional component (first 
            (q+1) columns) and D is the dimension of the linear component (last 
            D columns).
    
        data: (n, q+1+D)-array
            Euclidean coordinates of n directional-linear random sample points, 
            where (q+1) is the Euclidean dimension of the directional component 
            (first (q+1) columns) and D is the dimension of the linear component
            (last D columns).
       
        h: float
            Bandwidth parameter for the directional component. (Default: h=None. 
            Then a rule of thumb for directional KDEs with the von Mises kernel 
            in Garcia-Portugues (2013) is applied.)
            
        b: float
            Bandwidth parameter for the linear component. (Default: h=None. 
            Then the Silverman's rule of thumb is applied. See Chen et al.(2016) 
            for details.)
            
        q: int
            Intrinsic data dimension of directional components. (Default: q=2.)
            
        D: int
            Data dimension of linear components. (Default: D=1.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000.)
    
    Return:
        MS_new: (m,q+1+D)-array
            The collection of converged points yielded by the DirLinMS algorithm.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    assert q+1+D == D_t, "The dimension of the input data, "+str(D_t)+", should "\
    "equal to the sum of directional dimension ("+str(q+1)+") in its ambient "\
    "space and linear dimension ("+str(D)+")."
    
    data_Dir = data[:,:(q+1)]
    data_Lin = data[:,(q+1):(q+1+D)]

    # Rule of thumb for directional component
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
        if q == 2:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
                 (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
                                  (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
        print("The current bandwidth for directional component is " + str(h) + ".\n")
    
    # Rule of thumb for linear component
    if b is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
        print("The current bandwidth for linear component is "+ str(b) + ".\n")
    
    MS_new = np.copy(mesh_0)
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            # print('The directional-linear MS algorithm converges in '+ str(t-1) + ' steps!')
            break
        MS_old = np.copy(MS_new)
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_Dir = MS_old[i,:(q+1)]
                x_Lin = MS_old[i,(q+1):(q+1+D)]
                # Kernel weights
                ker_w_Dir = np.exp((np.dot(data_Dir, x_Dir) - 1)/ h**2)
                ker_w_Lin = np.exp(-np.sum(((x_Lin - data_Lin)/b)**2, axis=1)/2)
                # Mean shift updates for directional components
                x_Dir_new = np.sum(data_Dir * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1), 
                                   axis=0)
                x_Dir_new = x_Dir_new / LA.norm(x_Dir_new)
                MS_new[i,:(q+1)] = x_Dir_new
                # Recompute the directional kernel weights
                ker_w_Dir = np.exp((np.dot(x_Dir_new, data_Dir.T) - 1)/ h**2)
                # Mean shift updates for linear components
                x_Lin_new = np.sum(data_Lin * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1),
                                   axis=0) / np.sum(ker_w_Dir * ker_w_Lin)
                MS_new[i,(q+1):(q+1+D)] = x_Lin_new
                if LA.norm(MS_new - MS_old) <= eps:
                    conv_sign[i] = 1
            else:
                MS_new[i,:] = MS_old[i,:]
                
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    MS_new = MS_new[conv_sign == 1]
    # Drop NaN entries before returning the mode candidates
    return MS_new[~np.isnan(MS_new[:,0])]


@ray.remote
def DirLinSCMSLog_Ray(mesh_0, data, d=1, h=None, b=None, q=2, D=1, eps=1e-7, 
                      max_iter=1000, return_conv=False):
    '''
    Directional-linear Subspace Constrained Mean Shift Algorithm under the 
    log-density with the von Mises and Gaussian kernels (Simultaneous version)
    
    Parameters:
        mesh_0: (m, q+1+D)-array
            Eulidean coordinates of m directional-linear query points, where 
            (q+1) is the Euclidean dimension of the directional component (first 
            (q+1) columns) and D is the dimension of the linear component (last 
            D columns).
    
        data: (n, q+1+D)-array
            Euclidean coordinates of n directional-linear random sample points, 
            where (q+1) is the Euclidean dimension of the directional component 
            (first (q+1) columns) and D is the dimension of the linear component
            (last D columns).
       
        h: float
            Bandwidth parameter for the directional component. (Default: h=None. 
            Then a rule of thumb for directional KDEs with the von Mises kernel 
            in Garcia-Portugues (2013) is applied.)
            
        b: float
            Bandwidth parameter for the linear component. (Default: h=None. 
            Then the Silverman's rule of thumb is applied. See Chen et al.(2016) 
            for details.)
            
        q: int
            Intrinsic data dimension of directional components. (Default: q=2.)
            
        D: int
            Data dimension of linear components. (Default: D=1.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000.)
        
        return_conv: Boolean
            The indicator of whether the convergence signs for the initial points
            will be returned or not.
            
    Return:
        SCMS_new: (m,q+1+D)-array
            The collection of converged points yielded by the DirLinSCMS algorithm.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    assert q+1+D == D_t, "The dimension of the input data, "+str(D_t)+", should "\
    "equal to the sum of directional dimension ("+str(q+1)+") in its ambient "\
    "space and linear dimension ("+str(D)+")."
    
    data_Dir = data[:,:(q+1)]
    data_Lin = data[:,(q+1):(q+1+D)]

    # Rule of thumb for directional component
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
        if q == 2:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
                 (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
                                  (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
        print("The current bandwidth for directional component is " + str(h) + ".\n")
    
    # Rule of thumb for linear component
    if b is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
        print("The current bandwidth for linear component is "+ str(b) + ".\n")
    
    SCMS_new = np.copy(mesh_0)
    # Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))
    
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            # print('The directional-linear SCMS algorithm converges in ' + str(t-1) + ' steps!')
            break
        SCMS_old = np.copy(SCMS_new)
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_Dir = SCMS_old[i,:(q+1)]
                x_Lin = SCMS_old[i,(q+1):(q+1+D)]
                x_pts = SCMS_old[i,:]
                # Kernel weights
                ker_w_Dir = np.exp((np.dot(data_Dir, x_Dir) - 1)/ h**2)
                ker_w_Lin = np.exp(-np.sum(((x_Lin - data_Lin)/b)**2, axis=1)/2)
                # Compute the directional-linear KDE up to a constant
                den_prop = np.sum(ker_w_Dir * ker_w_Lin)
                if den_prop == 0:
                    # Set those points with zero density values to NaN
                    nan_arr = np.zeros_like(x_pts)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    # Compute the total gradient of the log density
                    tot_grad_Log_Dir = np.sum(data_Dir * ker_w_Dir.reshape(n,1) \
                                             * ker_w_Lin.reshape(n,1), axis=0)/((h**2)*den_prop)
                    tot_grad_Log_Lin = np.sum((data_Lin - x_Lin) * ker_w_Dir.reshape(n,1) \
                                               * ker_w_Lin.reshape(n,1), axis=0)/((b**2)*den_prop)
                    tot_grad_Log = np.concatenate([tot_grad_Log_Dir.reshape(q+1, 1), 
                                                   tot_grad_Log_Lin.reshape(D, 1)], axis=0)
                    # Compute the Hessian matrix of the log density
                    ## Hessian in the directional component: (q+1)*(q+1)
                    Log_Hess_Dir = np.dot(data_Dir.T, data_Dir * ker_w_Dir.reshape(n,1) \
                                         * ker_w_Lin.reshape(n,1))/((h**4)*den_prop) \
                        - np.dot(tot_grad_Log_Dir.reshape(q+1, 1), tot_grad_Log_Dir.reshape(1, q+1)) \
                        - np.eye(q+1) * np.sum(np.dot(data_Dir * ker_w_Dir.reshape(n,1) \
                                                      * ker_w_Lin.reshape(n,1), x_Dir))/((h**2)*den_prop)
                    x_Dir = x_Dir.reshape(q+1, 1)
                    proj_mat = np.eye(q+1) - np.dot(x_Dir, x_Dir.T)
                    Log_Hess_Dir = np.dot(np.dot(proj_mat, Log_Hess_Dir), proj_mat)
                    ## Hessian in the linear component: D*D
                    Log_Hess_Lin = np.dot((x_Lin - data_Lin).T, (x_Lin - data_Lin) \
                                * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1))/((b**4)*den_prop) \
                              - np.eye(D) * np.sum(ker_w_Dir.reshape(n,1) \
                                                   * ker_w_Lin.reshape(n,1))/((b**2)*den_prop) \
                              - np.dot(tot_grad_Log_Lin.reshape(D,1), tot_grad_Log_Lin.reshape(1,D))
                    ## Hessian in the off-diagonal part: (q+1)*D
                    x_Dir = x_Dir.reshape(q+1, )
                    Log_Hess_Off = np.dot(data_Dir.T, (data_Lin - x_Lin) * ker_w_Dir.reshape(n,1) \
                                       * ker_w_Lin.reshape(n,1))/((b**2)*(h**2)*den_prop) \
                             - np.dot(tot_grad_Log_Dir.reshape(q+1, 1), tot_grad_Log_Lin.reshape(1,D))
                    Log_Hess_Off = np.dot(proj_mat, Log_Hess_Off)
                    ## Concatenate to obtain the final Hessian
                    Log_Hess = np.zeros((q+1+D, q+1+D))
                    Log_Hess[:(q+1), :(q+1)] = Log_Hess_Dir
                    Log_Hess[:(q+1), (q+1):(q+1+D)] = Log_Hess_Off
                    Log_Hess[(q+1):(q+1+D), :(q+1)] = Log_Hess_Off.T
                    Log_Hess[(q+1):(q+1+D), (q+1):(q+1+D)] = Log_Hess_Lin
                    if (np.sum(np.isinf(Log_Hess)) > 0) or (np.sum(np.isnan(Log_Hess)) > 0):
                        # Set those points with infinite Hessians to NaN
                        nan_arr = np.zeros_like(x_pts)
                        nan_arr[:] = np.nan
                        conv_sign[i] = 1
                        x_new = nan_arr
                    else:
                        # Spectral decomposition
                        w, v = LA.eig(Log_Hess)
                        x_eig = np.concatenate([x_Dir.reshape(q+1, 1), 
                                                np.zeros((D,1))], axis=0)
                        # Obtain the eigenpairs within the tangent space
                        tang_eig_v = v[:, (abs(np.dot(x_eig.T, v)) < 1e-8)[0,:]]
                        tang_eig_w = w[(abs(np.dot(x_eig.T, v)) < 1e-8)[0,:]]
                        V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(q+D-d)]]
                        # Mean shift vector in directional and linear components
                        ms_Dir = (np.sum(data_Dir * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1), 
                                        axis=0) / np.sum(ker_w_Dir * ker_w_Lin)) * np.min([b/h, 1/(h**2)])
                        ms_Lin = (np.sum(data_Lin * ker_w_Dir.reshape(n,1) * ker_w_Lin.reshape(n,1),
                                        axis=0) / np.sum(ker_w_Dir * ker_w_Lin) - x_Lin) * np.min([h/b, 1/(b**2)])
                        ms_v = np.concatenate([ms_Dir.reshape(q+1, 1),
                                               ms_Lin.reshape(D, 1)], axis=0)
                        # Subspace constrained gradient and mean shift vector
                        SCMS_grad = np.dot(V_d, np.dot(V_d.T, tot_grad_Log))
                        # SCMS_grad_skew = np.dot(V_d, np.dot(V_d.T, tot_grad_Log_skew))
                        SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                        ## SCMS update
                        x_new = SCMS_v + x_pts.reshape(q+1+D, 1)
                        x_new = x_new.reshape(q+1+D, )
                        x_new[:(q+1)] = x_new[:(q+1)] / LA.norm(x_new[:(q+1)])
                        # if (LA.norm(SCMS_grad) < eps) or (LA.norm(SCMS_grad_skew) < eps):
                        if LA.norm(SCMS_grad) < eps:
                            conv_sign[i] = 1
                SCMS_new[i,:] = x_new
            else:
                SCMS_new[i,:] = SCMS_old[i,:]
    
    if t >= max_iter-1:
        print('The directional-linear SCMS algorithm reaches the maximum number of '\
               'iterations,'+str(max_iter)+', and has not yet converged.')
    if return_conv:
        conv_sign = conv_sign[~np.isnan(SCMS_new[:,0])]
        res = np.concatenate([SCMS_new[~np.isnan(SCMS_new[:,0])], conv_sign.reshape(-1,1)], axis=1)
        return res
    # Keep only the converged points
    SCMS_new = SCMS_new[conv_sign == 1]
    return SCMS_new[~np.isnan(SCMS_new[:,0])]
