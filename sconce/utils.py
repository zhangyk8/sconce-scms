# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: June 26, 2022

# Description: This script contains the utility functions for using our package 
# in practice.

import numpy as np
from numpy import linalg as LA

#================================================================================#

def cart2sph(x, y, z):
    '''
    Converting the Euclidean coordinate of a data point in R^3 to its Spherical 
    coordinates.
    
    Parameters
    ----------
        x, y, z: floats 
            Euclidean coordinate in R^3 of a data point.
    
    Returns
    ----------
        theta: float
            Longitude (ranging from -180 degree to 180 degree).
        phi: float
            Latitude (ranging from -90 degree to 90 degree).
        r: float 
            Radial distance from the origin to the data point.
    '''
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, dxy)
    theta, phi = np.rad2deg([theta, phi])
    return theta, phi, r

def sph2cart(theta, phi, r=1):
    '''
    Converting the Spherical coordinate of a data point to its Euclidean 
    coordinate in R^3.
    
    Parameters
    ----------
        theta: float
            Longitude (ranging from -180 degree to 180 degree).
        phi: float
            Latitude (ranging from -90 degree to 90 degree).
        r: float 
            Radial distance from the origin to the data point.
            
    Returns
    ----------
        x, y, z: floats 
            Euclidean coordinate in R^3 of a data point.
    '''
    theta, phi = np.deg2rad([theta, phi])
    z = r * np.sin(phi)
    rcosphi = r * np.cos(phi)
    x = rcosphi * np.cos(theta)
    y = rcosphi * np.sin(theta)
    return x, y, z

def CirSphSampling(N, lat_c=60, lon_range=[-180,180], sigma=0.01, 
                   pv_ax=np.array([0,0,1])):
    '''
    Generating data points from a circle on the unit sphere with additive 
    Gaussian noises to their Cartesian coordinates plus L2 normalizations.
    
    Parameters
    ----------
        N: int
            The number of randomly generated data points.
            
        lat_c: float
            The latitude of the circle with respect to the pivotal axis. 
            (range: 0-90)
            
        lon_range: 2-element list
            The longitude range that the circular structure covers. When 
            "lon_range=[-180,180]", the underlying structure is a full circle.
            
        sigma: float
            The standard deviation of Gaussian noises.
    
        pv_ax: (3,)-array
            The pivotal axis of the circle on the sphere from which the data 
            points are generated (plus noises).
            
    Return
    ----------
        pts_c_noise: (N,3)-array
            The Cartesian coordinates of N simulated data points.
    
    '''
    ## Random longitudes with range (0, 180)
    lon_c = np.random.rand(N,)*(lon_range[1]-lon_range[0]) + lon_range[0]
    lat_c = np.ones((N,))*lat_c
    x_c, y_c, z_c = sph2cart(lon_c, lat_c)

    pts_c = np.concatenate((x_c.reshape(len(x_c), 1), 
                            y_c.reshape(len(y_c), 1),
                            z_c.reshape(len(z_c), 1)), axis=1)
    ## Add Gaussian noises
    pts_c_noise = pts_c + sigma * np.random.randn(pts_c.shape[0], pts_c.shape[1])
    ## Standardize the noisy points
    pts_c_noise = pts_c_noise/np.sqrt(np.sum(pts_c_noise**2, axis=1)).reshape(N,1)
    
    ## Rotate the data samples accordingly
    mu_c = np.array([[0,0,1]])
    R = 2*np.dot(pv_ax.reshape(3,1)+mu_c.T, pv_ax.reshape(1,3)+mu_c)/\
        np.sum((mu_c+pv_ax.reshape(1,3))**2, axis=1) - np.identity(3)
    pts_c_noise = np.dot(R, pts_c_noise.T).T
    return pts_c_noise


def GaussMixture(N, mu=np.array([[1,1]]), cov=np.diag([1,1]).reshape(2,2,1), 
                 prob=[1.0]):
    '''
    Generating data points from a Gaussian mixture model.
    
    Parameters
    ----------
        N: int
            The number of randomly generated data points.
    
        mu: (m,d)-array
            The means of the Gaussian mixture model with m components.
       
        cov: (d,d,m)-array
            The (d,d)-covariance matrices of the Gaussian mixture model with 
            m components.
            
        prob: list of floats
            The mixture probabilities.
    
    Return
    ----------
        data_ps: (N,d)-array
            The Cartesian coordinates of N simulated data points.
    '''
    m = len(prob)   ## The number of mixtures
    d = mu.shape[1]  ## Dimension of the data
    assert (cov.shape[2] == len(prob)), "'cov.shape[2]' and 'len(prob)' "\
    "should be equal."
    inds = np.random.choice(list(range(m)), N, replace=True, 
                            p=np.array(prob)/sum(prob))
    data_ps = np.zeros((N,d))
    for i in range(m):
        data_ps[inds == i,:] = np.random.multivariate_normal(mu[i,:], cov[:,:,i], 
                                                             size=sum(inds == i))
    return data_ps


def vMFDensity(x, mu=np.array([[0,0,1]]), kappa=[1.0], prob=[1.0]):
    '''
    The q-dimensional von-Mises Fisher (vMF) density function or its mixture.
    
    Parameters
    ----------
        x: (n,d)-array
            The Eulidean coordinates of n query points on a unit hypersphere, 
            where d=q+1 is the Euclidean dimension of data.
    
        mu: (m,d)-array
            The Euclidean coordinates of the m mean directions for a mixture of 
            vMF densities.
       
        kappa: list of floats
            The concentration parameters for a mixture of vMF densities.
       
        prob: list of floats
            The mixture probabilities.
            
    Return
    --------
        mix_den: (n,)-array
            The corresponding density value on each query point.
    '''
    assert (mu.shape[1] == x.shape[1] and mu.shape[0] == len(prob)), \
    "The parameter 'x' and mu' should be a (n,d)-array and (m,d)-array, respectively, \
    and 'prob' should be a list of length m."
    assert (len(kappa) == len(prob)), "The parameters 'kappa' and 'prob' should \
    be of the same length."
    d = x.shape[1]   ## Euclidean dimension of the data
    prob = np.array(prob).reshape(len(prob), 1)
    kappa = np.array(kappa)
    dens = kappa**(d/2-1)*np.exp(kappa*np.dot(x, mu.T))/((2*np.pi)**(d/2)*sp.iv(d/2-1, kappa))
    mix_den = np.dot(dens, prob)
    return mix_den


def vMFSamp(n, mu=np.array([0,0,1]), kappa=1):
    '''
    Randomly sampling data points from a q-dimensional von-Mises Fisher (vMF) 
    density with analytic approaches.
    
    Parameters
    ---------
        n: int
            The number of sampling random data points.
        
        mu: (d, )-array
            The Euclidean coordinate of the mean directions of the q-dim vMF
            density, where d=q+1.
            
        kappa: float
            The concentration parameter of the vMF density.
    
    Return
    --------
        data_ps: (n, d)-array
            The Euclidean coordinates of the randomly sampled points from the 
            vMF density.
    '''
    
    d = len(mu)   # Euclidean dimension of the data
    if d == 3:    # when d=3 (S^2 case), no rejection sampling is required
        U1 = np.random.rand(n,1)
        U2 = 2*np.pi*np.random.rand(n,1)
        W = 1 + np.log(U1 + (1-U1)*np.exp(-2*kappa))/kappa
        X = np.cos(U2) * np.sqrt(1-W**2)
        Y = np.sin(U2) * np.sqrt(1-W**2)
        data_ps = np.concatenate((X,Y,W), axis=1)
    else:
        t1 = np.sqrt(4*(kappa**2) + (d-1)**2)
        b = (-2*kappa + t1)/(d-1)
        x0 = (1-b)/(1+b)
        m = (d-1)/2
        c = kappa*x0 + (d-1)*np.log(1-x0**2)
        data_ps = np.zeros((n,d))
        for i in range(n):
            t = -1000
            U = 1
            while t < np.log(U) + c:
                z = np.random.beta(a=m, b=m)
                U = np.random.rand(1,)[0]
                w = (1-(1+b)*z)/(1-(1-b)*z)
                t = kappa*w + (d-1)*np.log(1-x0*w)
            # Generate a vector from the uniform distribution on the unit hypersphere
            V = np.random.multivariate_normal(mean=np.zeros((d-1,)), 
                                              cov=np.identity(d-1), size=1)[0]
            V = V/np.sqrt(np.sum(V**2))
            data_ps[i, 0:(d-1)] = np.sqrt(1-w**2)*V
            data_ps[i, d-1] = w
    # Rotate the data samples accordingly
    mu_c = np.zeros((1,d))
    mu_c[0,d-1] = 1
    R = 2*np.dot(mu.reshape(d,1)+mu_c.T, mu.reshape(1,d)+mu_c) / \
        np.sum((mu_c+mu.reshape(1,d))**2, axis=1) - np.identity(d)
    data_ps = np.dot(R, data_ps.T).T
    return data_ps


def vMFRejectSamp(n, mu=np.array([0,0,1]), kappa=1):
    '''
    Randomly sampling data points from a q-dimensional von-Mises Fisher (vMF) 
    density via rejection sampling.
    
    Parameters
    ---------
        n: int
            The number of sampling random data points.
        
        mu: (d, )-array
            The Euclidean coordinate of the mean directions of the q-dim vMF
            density, where d=q+1.
            
        kappa: float
            The concentration parameter of the vMF density.
    
    Return
    --------
        data_ps: (n, d)-array
            The Euclidean coordinates of the randomly sampled points from the 
            vMF density.
    '''
    d = len(mu)   ## Euclidean dimension of the data
    data_ps = np.zeros((n,d))
    ## Sample points from standard normal and then standardize them
    sam_can = np.random.multivariate_normal(mean=np.zeros((d,)), cov=np.identity(d), size=n)
    dist_sam = np.sqrt(np.sum(sam_can**2, axis=1)).reshape(n,1)
    sam_can = sam_can/dist_sam

    unif_sam = np.random.uniform(0, 1, n)
    ## Reject some inadequate data points  
    ## (When the uniform proposal density is used, the normalizing constant in 
    ## front of the vMF density has no effects in rejection sampling.)
    mu = mu.reshape(d,1)
    sams = sam_can[unif_sam < np.exp(kappa*(np.dot(sam_can, mu)-1))[:,0],:]
    cnt = sams.shape[0]
    data_ps[:cnt,:] = sams
    while cnt < n:
        can_p = np.random.multivariate_normal(mean=np.zeros((d,)), cov=np.identity(d), size=1)
        can_p = can_p/np.sqrt(np.sum(can_p**2))
        unif_p = np.random.uniform(0, 1, 1)
        if np.exp(kappa*(np.dot(can_p, mu)-1)) > unif_p:
            data_ps[cnt,:] = can_p
            cnt += 1
    return data_ps


def vMFMixtureSamp(n, mu=np.array([[0,0,1]]), kappa=[1.0], prob=[1.0]):
    '''
    Randomly sampling data points from a mixture of q-dimensional von-Mises 
    Fisher (vMF) densities.
    
    Parameters
    ---------
        n: int
            The number of sampling random data points.
    
        mu: (m,d)-array
            The Euclidean coordinates of the m mean directions for a mixture of 
            vMF densities.
       
        kappa: list of floats
            The concentration parameters for the mixture of von-Mises Fisher 
            densities.
       
        prob: list of floats
            The mixture probabilities.
            
    Return
    --------
        data_ps: (n,d)-array
            The Euclidean coordinates of the randomly sampled points from the 
            vMF mixture.
    '''
    m = len(prob)   ## The number of mixtures
    d = mu.shape[1]  ## Euclidean dimension of the data
    assert (len(kappa) == len(prob)), "The parameters 'kappa' and 'prob' should be of the same length."
    inds = np.random.choice(list(range(m)), n, replace=True, p=np.array(prob)/sum(prob))
    data_ps = np.zeros((n,d))
    for i in range(m):
        data_ps[inds == i,:] = vMFSamp(sum(inds == i), mu=mu[i,:], kappa=kappa[i])
    return data_ps


def SmoothBootstrap_vMF(data, B=1000, h=None):
    '''
    Resampling a dataset using the smoothed bootstrap with the von Mises kernel.
    
    Parameters
    ---------
        data: (n,d)-array
            The Euclidean coordinates of n directional data points in 
            the d-dimensional Euclidean space, where d=q+1.
            
        B: int
            The number of bootstrapping times.
            
        h: float
           The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
           directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
           is applied.)
            
    Return
    --------
        data_Boot: (n,d)-array
            The Euclidean coordinates of the smoothed bootstrap dataset.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Euclidean Dimension of the data

    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (D - R_bar ** 2) / (1 - R_bar ** 2)
        if D == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(D / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (D / 2) * (2 * (D - 1) * sp.iv(D/2, 2*kap_hat) + \
                                  (D+1) * kap_hat * sp.iv(D/2+1, 2*kap_hat)))) ** (1/(D + 3))
        print("The current bandwidth is " + str(h) + ".\n")
        
    data_Boot = np.zeros((B, d))
    for i in range(B):
        ind = np.random.choice(n, size=1, replace=True)
        data_Boot[i,:] = vMFSamp(1, mu=data[ind[0],:], kappa=1/(h**2))
    return data_Boot