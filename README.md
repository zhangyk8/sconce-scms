[![PyPI pyversions](https://img.shields.io/pypi/pyversions/sconce-scms.svg)](https://pypi.python.org/pypi/sconce-scms/)
[![PyPI version](https://badge.fury.io/py/sconce-scms.svg)](https://badge.fury.io/py/sconce-scms)
[![PyPI Downloads](https://pepy.tech/badge/sconce-scms)](https://pepy.tech/project/sconce-scms)
[![Documentation Status](https://readthedocs.org/projects/sconce-scms/badge/?version=latest)](http://sconce-scms.readthedocs.io/?badge=latest)

# SCONCE-SCMS
## Spherical and Conic Cosmic Web Finders with Extended SCMS Algorithms

The subspace consrained mean shift (SCMS) algorithm `[[1]]{#sconce}`

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

Quick Start
--------

References
--------

[[1]]{#sconce} Y. Zhang, R. S. de Souza, and Y.-C. Chen (2022+). SCONCE: A Filament Finder for Spherical and Conic Cosmic Web Geometries.

[2] U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.

[3] Y.-C. Chen, S. Ho, P. E. Freeman, C. R. Genovese, and L. Wasserman (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156.

[4] Y. Zhang and Y.-C. Chen (2021). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. *Journal of Machine Learning Research*, **22**(154), 1-92.

[5] Y. Zhang and Y.-C. Chen (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, [https://doi.org/10.1093/imaiai/iaac005](https://doi.org/10.1093/imaiai/iaac005).
