Sperical and Conic Cosmic Web Finder in Python
===================================

**SCONCE-SCMS** (**S**\pherical and **CON**\ic **C**\osmic w\ **E**\b finder with the extended **SCMS** algorithms [1]_) is a Python library for detecting the cosmic web structures (primarily cosmic filaments and the associated cosmic nodes) from a collection of discrete observations with the extended subspace constrained mean shift (SCMS) algorithms on the 2D (RA,DEC) celestial sphere :math:`\mathbb{S}^2` or the 3D (RA,DEC,redshift) light cone :math:`\mathbb{S}^2\times\mathbb{R}`. 

(Notes: RA -- Right Ascension, i.e., the celestial longitude; DEC -- Declination, i.e., the celestial latitude.)

A quick introduction to the methodology
--------------

The subspace constrained mean shift (SCMS) algorithm [2]_ is a gradient ascent typed method dealing with the estimation of local principal curves, more widely known as density ridges in statistics [3]_. The one-dimensional density ridge traces over the curves where observational data are highly concentrated and thus serves as a natural model for cosmic filaments in our Universe [4]_. One advantage of modeling cosmic filaments as density ridges is that they can be efficiently estimated by the kernel density estimator (KDE) and the subsequent SCMS algorithm in a statistically consistent way.

Whereas the standard SCMS algorithm [2]_ is well-suited for identifying the density ridges in any "flat" Euclidean space :math:`\mathbb{R}^D`, it exhibits large bias in estimating the density ridges on the data space with a non-linear curvature; see examples in [1]_ and [5]_. In astronomy, however, one often encounters observations from the 2D (RA,DEC) celestial sphere :math:`\mathbb{S}^2` or the 3D (RA,DEC,redshift) light cone :math:`\mathbb{S}^2\times\mathbb{R}`. To resolve the estimation bias of the standard SCMS algorithm on these two data spaces, we propose our extended SCMS algorithms (*DirSCMS* [5]_ and *DirLinSCMS* [6]_ methods) that are adaptive to the spherical and conic geometries, respectively. At a high level, we utilize the directional or directional-linear KDEs to estimate the underlying density function and carefully design the iteration formulae for our extended SCMS algorithms. 

More details can be found in :doc:`Methodology <methods>` and the reference paper [1]_.


.. note::

   This project is under active development.
   

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   methods
   Example_SCMS
   api_reference
   

How to Cite SCONCE-SCMS
---------

If you use ``sconce-scms`` in your research, please cite the following papers:

* "Y. Zhang, R. S. de Souza, and Y.-C. Chen (2022). SCONCE: A cosmic web finder for spherical and conic geometries *arXiv preprint arXiv:2207.07001*."
* "Y.-C. Chen, S. Ho, P. E. Freeman, C. R. Genovese, and L. Wasserman (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156."

   
References
----------
.. [1] Zhang, Y., de Souza, R. S., and Chen, Y.-C. (2022). SCONCE: A cosmic web finder for spherical and conic geometries *arXiv preprint arXiv:2207.07001*
.. [2] Ozertem, U. and Erdogmus, D. (2011). Locally defined principal curves and surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.
.. [3] Genovese, C.R., Perone-Pacifico, M., Verdinelli, I. and Wasserman, L. (2014). Nonparametric ridge estimation. The Annals of Statistics, **42**(4), 1511-1545.
.. [4] Chen, Y.-C., Ho, S., Freeman, P.E., Genovese, C.R. and Wasserman, L. (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156.
.. [5] Zhang, Y. and Chen, Y.-C. (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, `https://doi.org/10.1093/imaiai/iaac005 <https://doi.org/10.1093/imaiai/iaac005>`_.
.. [6] Zhang, Y. and Chen, Y.-C. (2021). Mode and ridge estimation in euclidean and directional product spaces: A mean shift approach. *arXiv preprint arXiv:2110.08505*, `https://arxiv.org/abs/2110.08505 <https://arxiv.org/abs/2110.08505>`_.
