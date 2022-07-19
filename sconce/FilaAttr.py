# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: July 03, 2022

# Description: This script contains code for detecting the knots (or intersection 
# points) and computing the length of a given filament/web-like structure.

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from numpy import linalg as LA

def DetectKnot(x, r_in, r_out, fila_map):
    '''
    Detecting if a point on the filament is a knot/intersection 
    (Metric Graph Method).
    
    Parameters
    ----------
        x: (D,)-array
            The coordinate of the point of interest.
        
        r_in: float
            The radius of the inner ball around the point of interest.
        
        r_out: float
            The radius of the outer ball around the point of interest.
        
        fila_map : (N,D)-array
            The coordinates of all the filament points.

    Returns
    -------
        'Knot'/'Non-Knot': str
            The indicator of whether the point of interest is a knot/intersection
            on the input filament.

    '''
    dist = LA.norm(fila_map - x.values, axis=1)
    can_pts = fila_map[(dist >= r_in) & (dist <= r_out)]
    if can_pts.shape[0] < 3:
        return 'Non-Knot'
    clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', 
                                         linkage='average', distance_threshold=(r_in + r_out)/2)
    clustering.fit_predict(can_pts)
    if len(np.unique(clustering.labels_)) >= 3:
        return 'Knot'
    else:
        return 'Non-Knot'
    
