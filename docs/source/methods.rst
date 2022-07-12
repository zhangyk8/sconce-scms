Methodology
===========

Standard SCMS Algorithm on the Euclidean Space :math:`\mathbb{R}^D`
------------

The subspace constrained mean shift (SCMS) algorithm [2]_ is designed to estimate the density ridges, the lower dimensional structures at which the density peak. Given a (smooth) density function :math:`p` supported on :math:`\mathbb{R}^D`, its d-dimensional density ridge is defined as

.. math::

    R_d(p) = \left\{\mathbf{x} \in \mathbb{R}^D: V_E(\mathbf{x})^T \nabla p(\mathbf{x})=\mathbf{0} \right\},
    
where :math:`\nabla p(\mathbf{x})` is the gradient of :math:`p` and :math:`V_E(\mathbf{x})=\left[\mathbf{v}_{d+1}(\mathbf{x}),..., \mathbf{v}_D(\mathbf{x})\right] \in \mathbb{R}^{D\times (D-d)}` consists of the last :math:`(D-d)` eigenvectors of the Hessian :math:`\nabla\nabla p(\mathbf{x})` associated with a descending order of eigenvalues :math:`\lambda_{d+1}(\mathbf{x}) \geq \cdots \geq \lambda_D(\mathbf{x})`. Under the scenario of cosmic filament detection in the flat Euclidean space :math:`\mathbb{R}^D`, the one-dimensional density ridge :math:`R_1(p)` serve as an ideal choice of the theoretical cosmic filament model.

To estimate the theoretical density ridge :math:`R_d(p)` in practice, we leverage the plug-in estimator :math:`R_d(\widehat{p})` derived from the kernel density estimator :math:`\widehat{p}` (KDE) as:

.. math::

    \widehat{p}(\mathbf{x}) = \frac{1}{nb^D} \sum_{i=1}^n K\left(\left\|\frac{\mathbf{x}-\mathbf{X}_i}{b} \right\|_2^2 \right),

where :math:`\{\mathbf{X}_1,...,\mathbf{X}_n\} \subset \mathbb{R}^D` is a random sample from :math:`p`, :math:`K:\mathbb{R} \to \mathbb{R}^+` is the kernel function (e.g., the Gaussian kernel :math:`K(r)=\frac{1}{(2\pi)^{D/2}} \exp\left(\frac{r}{2} \right)`), and :math:`b` is the smoothing bandwidth parameter. The standard SCMS algorithm in :math:`\mathbb{R}^D` is then applied to an initial mesh of points and iterates the following formula for :math:`t=0,1,...`:

.. math::

    \mathbf{x}^{(t+1)} \gets \mathbf{x}^{(t)} + \widehat{V}_E(\mathbf{x}^{(t)}) \widehat{V}_E(\mathbf{x}^{(t)})^T \left[ \frac{\sum_{i=1}^n \mathbf{X}_i K'\left(\left\|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{b}\right\|_2^2 \right)}{\sum_{i=1}^n K'\left(\left\|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{b}\right\|_2^2 \right)} - \mathbf{x}^{(t)} \right]

until convergence on each initial point :math:`\mathbf{x}^{(0)}`, where :math:`\widehat{V}_E=\left[\widehat{\mathbf{v}}_{d+1}(\mathbf{x}),..., \widehat{\mathbf{v}}_D(\mathbf{x})\right] \in \mathbb{R}^{D\times (D-d)}` has its columns as the last :math:`(D-d)` eigenvectors of the estimated Hessian :math:`\nabla\nabla \hat{p}(\mathbf{x})`. The set of converged points is a discrete sample from the estimated density ridge :math:`R_d(\widehat{p})`.

Despite its fast convergence and wide applications in identifying density ridges, the standard SCMS algorithm fails to take into account any non-linear curvature in the data space; see Figure 1 in [1]_ and Appendix B in [5]_ for the spherical data case. 



References
----------

.. [1] Zhang, Y., de Souza, R. S., and Chen, Y.-C. (2022+). SCONCE: A Cosmic Web Finder for Spherical and Conic Geometries.
.. [2] Ozertem, U. and Erdogmus, D. (2011). Locally defined principal curves and surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.
.. [3] Genovese, C.R., Perone-Pacifico, M., Verdinelli, I. and Wasserman, L. (2014). Nonparametric ridge estimation. The Annals of Statistics, **42**(4), 1511-1545.
.. [4] Chen, Y.-C., Ho, S., Freeman, P.E., Genovese, C.R. and Wasserman, L. (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156.
.. [5] Zhang, Y. and Chen, Y.-C. (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, `https://doi.org/10.1093/imaiai/iaac005 <https://doi.org/10.1093/imaiai/iaac005>`_.
.. [6] Zhang, Y. and Chen, Y.-C. (2021). Mode and ridge estimation in euclidean and directional product spaces: A mean shift approach. *arXiv preprint arXiv:2110.08505*, `https://arxiv.org/abs/2110.08505 <https://arxiv.org/abs/2110.08505>`_.
