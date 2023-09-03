[![PyPI pyversions](https://img.shields.io/pypi/pyversions/sconce-scms.svg)](https://pypi.python.org/pypi/sconce-scms/)
[![PyPI version](https://badge.fury.io/py/sconce-scms.svg)](https://badge.fury.io/py/sconce-scms)
[![Downloads](https://static.pepy.tech/badge/sconce-scms)](https://pepy.tech/project/sconce-scms)
[![Documentation Status](https://readthedocs.org/projects/sconce-scms/badge/?version=latest)](http://sconce-scms.readthedocs.io/?badge=latest)

# SCONCE-SCMS
## Spherical and Conic Cosmic Web Finders with the Extended SCMS Algorithms

<img src="/SCONCE_Sconce_Logo.png" alt="sconce_logo" width="500"/>

**SCONCE-SCMS** (**S**pherical and **CON**ic **C**osmic w **E**b finder with the extended **SCMS** algorithms [[1]](#sconce) is a Python library for detecting the cosmic web structures (primarily cosmic filaments and the associated cosmic nodes) from a collection of discrete observations with the extended subspace constrained mean shift (SCMS) algorithms ([[2]](#scms), [[5]](#dirscms), [[6]](#dirlinscms)) on the unit (hyper)sphere (_in most cases, the 2D (RA,DEC) celestial sphere_), and the directional-linear product space (_most commonly, the 3D (RA,DEC,redshift) light cone_). 

(Notes: RA -- Right Ascension, i.e., the celestial longitude; DEC -- Declination, i.e., the celestial latitude.)

* Free software: MIT license
* Documentation: https://sconce-scms.readthedocs.io.


Installation guide
--------

```sconce-scms``` requires Python 3.6+ (earlier version might be applicable), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), and [Ray](https://ray.io/) (optional and only used for parallel computing). To install the latest version of ```sconce-scms``` from this repository, run:

```
python setup.py install
```

To pip install a stable release, run:
```
pip install sconce-scms
```

References
--------

<a name="sconce">[1]</a> Y. Zhang, R. S. de Souza, and Y.-C. Chen (2022). SCONCE: A cosmic web finder for spherical and conic geometries. *Monthly Notices of the Royal Astronomical Society*, **517** (1): 1197–1217.

<a name="scms">[2]</a> U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.

[3] Y.-C. Chen, S. Ho, P. E. Freeman, C. R. Genovese, and L. Wasserman (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454** (1), 1140-1156.

[4] Y. Zhang and Y.-C. Chen (2021). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. *Journal of Machine Learning Research*, **22** (154), 1-92.

<a name="dirscms">[5]</a> Y. Zhang and Y.-C. Chen (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, [https://doi.org/10.1093/imaiai/iaac005](https://doi.org/10.1093/imaiai/iaac005).

<a name="dirlinscms">[6]</a> Y. Zhang and Y.-C. Chen (2021). Mode and ridge estimation in euclidean and directional product spaces: A mean shift approach. *arXiv preprint arXiv:2110.08505*.
