#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: June 22, 2022

Description: This script contains the utility functions for using our package 
in practice.
"""

import numpy as np

#==========================================================================================#

def cart2sph(x, y, z):
    '''
    Converting the Euclidean coordinate of a data point in R^3 to its Spherical 
    coordinates.
    
    Parameters:
        x, y, z: floats 
            Euclidean coordinate in R^3 of a data point.
    
    Returns:
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
    
    Parameters:
        theta: float
            Longitude (ranging from -180 degree to 180 degree).
        phi: float
            Latitude (ranging from -90 degree to 90 degree).
        r: float 
            Radial distance from the origin to the data point.
            
    Returns:
        x, y, z: floats 
            Euclidean coordinate in R^3 of a data point.
    '''
    theta, phi = np.deg2rad([theta, phi])
    z = r * np.sin(phi)
    rcosphi = r * np.cos(phi)
    x = rcosphi * np.cos(theta)
    y = rcosphi * np.sin(theta)
    return x, y, z