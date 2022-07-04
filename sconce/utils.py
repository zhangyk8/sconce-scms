import numpy as np


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
            
        lat_c: float (range: 0-90)
            The latitude of the circle with respect to the pivotal axis.
            
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
