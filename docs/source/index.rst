Sperical and Conic Cosmic Web Finder in Python
===================================

**SCONCE-SCMS** (**S**\pherical and **CON**\ic **C**\osmic w\ **E**\b finder with the extended **SCMS** algorithms [1]_) is a Python library for detecting the cosmic web structures (primarily cosmic filaments and the associated cosmic nodes) from a collection of discrete observations with the extended subspace constrained mean shift (SCMS) algorithms on the 2D (RA,DEC) celestial sphere :math:`\mathbb{S}^2` or 3D (RA,DEC,redshift) light cone :math:`\mathbb{S}^2\times\mathbb{R}`.

Standard SCMS Algorithm on the Euclidean Space :math:`\mathbb{R}^D`
------------

The subspace constrained mean shift (SCMS) algorithm [2]_ is a gradient ascent typed method that dealing with the estimation of local principal curves, more widely known as density ridges in statistics [3]_. Given a (smooth) density function :math:`p` in :math:`\mathbb{R}^D`, its d-dimensional density ridge is defined as

.. math::

    R_d(p) = \left\{\mathbf{x} \in \mathbb{R}^D: V_E(\mathbf{x})^T \nabla p(\mathbf{x})=\mathbf{0} \right\},
    
where :math:`\nabla p(\mathbf{x})` is the gradient of :math:`p` and :math:`V_E(\mathbf{x})=\left[\mathbf{v}_{d+1}(\mathbf{x}),..., \mathbf{v}_D(\mathbf{x})\right] \in \mathbb{R}^{D\times (D-d)}` consists of the last :math:`(D-d)` eigenvectors of the Hessian :math:`\nabla\nabla p(\mathbf{x})` associated with a descending order of eigenvalues :math:`\lambda_{d+1}(\mathbf{x}) \geq \cdots \geq \lambda_D(\mathbf{x})`. Under the scenario of cosmic filament detection in the flat Euclidean space :math:`\mathbb{R}^D`, the one-dimensional density ridge :math:`R_1(p)` serve as an ideal choice of the theoretical cosmic filament model.

To estimate the theoretical density ridge :math:`R_d(p)` in practice, one leverages the SCMS algorithm. It is built on top of  



.. note::

   This project is under active development.
   

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   api_reference
   
   
References
----------
.. [1] Y. Zhang, R. S. de Souza, and Y.-C. Chen (2022+). SCONCE: A Filament Finder for Spherical and Conic Cosmic Web Geometries.
.. [2] U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.
.. [3] C. R. Genovese, M. Perone-Pacifico, I. Verdinelli, and L. Wasserman (2014). Nonparametric ridge estimation. The Annals of Statistics, **42**(4), 1511-1545.
.. [4] Y.-C. Chen, S. Ho, P. E. Freeman, C. R. Genovese, and L. Wasserman (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156.
.. [5] Y. Zhang and Y.-C. Chen (2021). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. *Journal of Machine Learning Research*, **22**(154), 1-92.
.. [6] Y. Zhang and Y.-C. Chen (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, `https://doi.org/10.1093/imaiai/iaac005 <https://doi.org/10.1093/imaiai/iaac005>`_.
