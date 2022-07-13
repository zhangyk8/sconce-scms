Methodology
===========

Standard SCMS Algorithm on the Euclidean Space :math:`\mathbb{R}^D`
------------

The subspace constrained mean shift (SCMS) algorithm [2]_ is designed to estimate the density ridges, the lower dimensional structures at which the density peak. Given a (smooth) density function :math:`p` supported on :math:`\mathbb{R}^D`, its :math:`d`-dimensional density ridge is defined as [3]_

.. math::

    R_d(p) = \left\{\mathbf{x} \in \mathbb{R}^D: V_E(\mathbf{x})^T \nabla p(\mathbf{x})=\mathbf{0}, \lambda_{d+1}(\mathbf{x}) < 0 \right\},
    
where :math:`\nabla p(\mathbf{x})` is the gradient of :math:`p` and :math:`V_E(\mathbf{x})=\left[\mathbf{v}_{d+1}(\mathbf{x}),..., \mathbf{v}_D(\mathbf{x})\right] \in \mathbb{R}^{D\times (D-d)}` consists of the last :math:`(D-d)` eigenvectors of the Hessian :math:`\nabla\nabla p(\mathbf{x})` associated with a descending order of eigenvalues :math:`\lambda_{d+1}(\mathbf{x}) \geq \cdots \geq \lambda_D(\mathbf{x})`. Under the scenario of cosmic filament detection in the flat Euclidean space :math:`\mathbb{R}^D`, the one-dimensional density ridge :math:`R_1(p)` serve as an ideal choice of the theoretical cosmic filament model.

To estimate the theoretical density ridge :math:`R_d(p)` in practice, we leverage the plug-in estimator :math:`R_d(\widehat{p})` derived from the kernel density estimator :math:`\widehat{p}` (KDE) as:

.. math::

    \widehat{p}(\mathbf{x}) = \frac{1}{nb^D} \sum_{i=1}^n K\left(\left\|\frac{\mathbf{x}-\mathbf{X}_i}{b} \right\|_2^2 \right),

where :math:`\{\mathbf{X}_1,...,\mathbf{X}_n\} \subset \mathbb{R}^D` is a random sample from :math:`p`, :math:`\|\cdot\|_2` is the usual Euclidean norm, :math:`K:\mathbb{R} \to \mathbb{R}^+` is the kernel function (e.g., the Gaussian kernel :math:`K(r)=\frac{1}{(2\pi)^{D/2}} \exp\left(\frac{r}{2} \right)`), and :math:`b` is the smoothing bandwidth parameter. The standard SCMS algorithm in :math:`\mathbb{R}^D` is then applied to an initial mesh of points and iterates the following formula for :math:`t=0,1,...`:

.. math::

    \mathbf{x}^{(t+1)} \gets \mathbf{x}^{(t)} + \widehat{V}_E(\mathbf{x}^{(t)}) \widehat{V}_E(\mathbf{x}^{(t)})^T \left[ \frac{\sum_{i=1}^n \mathbf{X}_i K'\left(\left\|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{b}\right\|_2^2 \right)}{\sum_{i=1}^n K'\left(\left\|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{b}\right\|_2^2 \right)} - \mathbf{x}^{(t)} \right]

until convergence on each initial point :math:`\mathbf{x}^{(0)}`, where :math:`\widehat{V}_E=\left[\widehat{\mathbf{v}}_{d+1}(\mathbf{x}),..., \widehat{\mathbf{v}}_D(\mathbf{x})\right] \in \mathbb{R}^{D\times (D-d)}` has its columns as the last :math:`(D-d)` eigenvectors of the estimated Hessian :math:`\nabla\nabla \hat{p}(\mathbf{x})`. The set of converged points is a discrete sample from the estimated density ridge :math:`R_d(\widehat{p})`.

**Despite its fast convergence and wide applications in identifying density ridges, the standard SCMS algorithm fails to take into account any non-linear curvature in the data space**; see Figure 1 in [1]_ and Appendix B in [5]_ for the spherical data case. 


Directional SCMS (*DirSCMS*) Algorithm on the Unit Sphere :math:`\mathbb{S}^q`
------------

Our *DirSCMS* algorithm [5]_ takes a discrete collection of observations :math:`\{\mathbf{X}_1,...,\mathbf{X}_n\}` on the unit (hyper)sphere :math:`\mathbb{S}^q=\left\{\mathbf{x}\in \mathbb{R}^{q+1}:\|\mathbf{x}\|=1 \right\}` and estimates the density ridge of the underlying directional density function :math:`f:\mathbb{S}^q \to \mathbb{R}`. The definition of the order-:math:`d` directional density ridge of :math:`f` is generalized from the above Euclidean counterpart as:

.. math::

    R_d(f) = \left\{\mathbf{x} \in \mathbb{S}^q: V_D(\mathbf{x})^T \mathtt{grad} f(\mathbf{x})=\mathbf{0}, \lambda_{d+1}(\mathbf{x}) < 0 \right\},
    
where :math:`\mathtt{grad} f(\mathbf{x})` is the Riemannian gradient of :math:`f` and :math:`V_D(\mathbf{x})=\left[\mathbf{v}_{d+1}(\mathbf{x}),..., \mathbf{v}_q(\mathbf{x})\right] \in \mathbb{R}^{(q+1)\times (q-d)}` consists of the last :math:`(q-d)` eigenvectors of the Riemannian Hessian :math:`\mathcal{H} f(\mathbf{x})` within the tangent space of :math:`\mathbb{S}^q` at :math:`\mathbf{x}` associated with a descending order of eigenvalues :math:`\lambda_{d+1}(\mathbf{x}) \geq \cdots \geq \lambda_q(\mathbf{x})`. Notice that :math:`\mathcal{H} f(\mathbf{x})` has :math:`q` orthonormal eigenvectors spanning the tangent space of :math:`\mathbb{S}^q` at :math:`\mathbf{x}` and another unit eigenvector :math:`\mathbf{x}` orthogonal to the tangent space. The directional density ridge :math:`R_d(f)` is indeed adaptive to the spherical geometry of :math:`\mathbb{S}^q` because it is defined through the Riemannian gradient and Hessian of the directional density :math:`f` within the tangent space of :math:`\mathbb{S}^q`. To compute these derivative quantities, one can extend the domain of :math:`f` from :math:`\mathbb{S}^q` to its ambient Euclidean space :math:`\mathbb{R}^{q+1}\setminus\{\mathbf{0}\}`. Then, the Riemannian gradient and Hessian on :math:`\mathbb{S}^q` are connected with the total gradient :math:`\nabla f(\mathbf{x})` and Hessian :math:`\mathcal{H} f(\mathbf{x})` in :math:`\mathbb{R}^{q+1}` as:

.. math::

    \mathtt{grad} f(\mathbf{x}) = (\mathbf{I}_{q+1} -\mathbf{x}\mathbf{x}^T) \nabla f(\mathbf{x}),
    
.. math::

    \mathcal{H} f(\mathbf{x}) = (\mathbf{I}_{q+1} -\mathbf{x}\mathbf{x}^T) \left[\nabla\nabla f(\mathbf{x}) - \nabla f(\mathbf{x})^T \mathbf{x} \cdot \mathbf{I}_{q+1} \right] (\mathbf{I}_{q+1} -\mathbf{x}\mathbf{x}^T),
    
where :math:`\mathbf{I}_{q+1}\in \mathbb{R}^{(q+1)\times (q+1)}` is the identity matrix. In the application of modeling cosmic filaments on the celestial sphere, one can take :math:`q=2` and :math:`d=1` in the above definition. 


To identify the directional density ridge :math:`R_d(f)` from :math:`\{\mathbf{X}_1,...,\mathbf{X}_n\} \subset \mathbb{S}^q`, we first estimate the directional density :math:`f` via the directional KDE ([6]_, [7]_) as:

.. math::

    \widehat{f}_b(\mathbf{x}) = \frac{C_L(b)}{n} \sum_{i=1}^n L\left(\frac{1-\mathbf{x}^T\mathbf{X}_i}{b^2} \right),
    
where :math:`L(\cdot)` is the directional kernel (e.g., the von Mises kernel :math:`L(r)=e^{-r}`), :math:`b` is the smoothing bandwidth parameter, and :math:`C_L(b)` is the normalizing constant ensuring that :math:`\widehat{f}_b` is a valid density on :math:`\mathbb{S}^q`. Again, the estimated density ridge :math:`R_d(\widehat{f}_b)` based on the directional KDE :math:`\widehat{f}_b` is a statistically consistent estimator of the theoretical density ridge :math:`R_d(f)` and can be practically identified by our *DirSCMS* algorithm with its iterative formula as:

.. math::

    \mathbf{x}^{(t+1)} \gets \mathbf{x}^{(t)} - \widehat{V}_D(\mathbf{x}^{(t)}) \widehat{V}_D(\mathbf{x}^{(t)})^T \left[\frac{\sum_{i=1}^n \mathbf{X}_i L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b^2} \right)}{\left\|\sum_{i=1}^n \mathbf{X}_i L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b^2} \right) \right\|_2} \right]
    
.. math::

    \text{ and } \quad \mathbf{x}^{(t+1)} \gets \frac{\mathbf{x}^{(t+1)}}{\left\| \mathbf{x}^{(t+1)} \right\|_2}

for :math:`t=0,1,...`, where :math:`\widehat{V}_D(\mathbf{x}) = \left[\widehat{\mathbf{v}}_{d+1}(\mathbf{x}),..., \widehat{\mathbf{v}}_q(\mathbf{x}) \right] \in \mathbb{R}^{(q+1)\times (q-d)}` has its columns as the last :math:`(q-d)` eigenvectors of the estimated Riemannian Hessian :math:`\mathcal{H} \widehat{f}_b(\mathbf{x})` within the tangent space of :math:`\mathbb{S}^q` at :math:`\mathbf{x}`. Notice that we leverage the normalized (total) gradient

.. math::

    \frac{\nabla \hat{f}_b(\mathbf{x}^{(t)})}{\left\|\nabla \hat{f}_b(\mathbf{x}^{(t)}) \right\|_2} = \frac{\sum_{i=1}^n \mathbf{X}_i L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b^2} \right)}{\left\|\sum_{i=1}^n \mathbf{X}_i L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b^2} \right) \right\|_2}
    
in the design of our *DirSCMS* algorithm in pursuit of a faster convergence rate [5]_.



Directional-linear SCMS (*DirLinSCMS*) Algorithm on the 3D Light Cone :math:`\mathbb{S}^2\times \mathbb{R}`
------------

Our *DirLinSCMS* algorithm [8]_ makes a further generalization of the above *DirSCMS* algorithm and addresses the density ridge estimation problem on a directional-linear product space :math:`\mathbb{S}^q\times \mathbb{R}^D`. (The implementation of the *DirLinSCMS* algorithm in our ``sconce-scms`` library *does not* restrict to the 3D light cone but accommodates this general form :math:`\mathbb{S}^q\times \mathbb{R}^D` of the directional-linear space.) We assume that its input data comprise independent and identically distributed (i.i.d.) observations :math:`(\mathbf{X}_i,\mathbf{Z}_i) \in \mathbb{S}^q\times \mathbb{R}^D, i=1,...,n` sampled from a directional-linear density :math:`f_{dl}(\mathbf{x},\mathbf{z})`. The theoretical density ridge is defined similarly as:

.. math::

    R_d(f_{dl}) = \left\{(\mathbf{x},\mathbf{z}) \in \mathbb{S}^q \times \mathbb{R}^D: V_{dl}(\mathbf{x},\mathbf{z})^T \mathtt{grad} f_{dl}(\mathbf{x},\mathbf{z})=\mathbf{0}, \lambda_{d+1}(\mathbf{x},\mathbf{z}) < 0 \right\},
    
where :math:`\mathtt{grad} f_{dl}(\mathbf{x},\mathbf{z})` is the Riemannian gradient of :math:`f_{dl}` and :math:`V_{dl}(\mathbf{x},\mathbf{z})=\left[\mathbf{v}_{d+1}(\mathbf{x},\mathbf{z}),..., \mathbf{v}_{q+D}(\mathbf{x},\mathbf{z})\right] \in \mathbb{R}^{(q+1+D)\times (q+D-d)}` consists of the last :math:`(q+D-d)` eigenvectors of the Riemannian Hessian :math:`\mathcal{H} f_{dl}(\mathbf{x},\mathbf{z})` within the tangent space of :math:`\mathbb{S}^q \times \mathbb{R}^D` at :math:`(\mathbf{x},\mathbf{z})` (equivalently, the orthogonal space of :math:`(\mathbf{x},\mathbf{0})` in :math:`\mathbb{R}^{q+1+D}`) associated with a descending order of eigenvalues :math:`\lambda_{d+1}(\mathbf{x},\mathbf{z}) \geq \cdots \geq \lambda_{q+D}(\mathbf{x},\mathbf{z})`. The Riemannian gradient and Hessian of :math:`f_{dl}` can also be expressed in terms of its total gradient and Hessian in the ambient Euclidean space :math:`\mathbb{R}^{q+1+D}`; see, e.g., Appendix A in [1]_.

Analogously, the underlying density :math:`f_{dl}` and its density ridge can be estimated by directional-linear KDE [9]_ as:

.. math::

    \widehat{f}_{dl}(\mathbf{x},\mathbf{z}) = \frac{C_L(b_1)}{nb_2^D} \sum_{i=1}^n L\left(\frac{1-\mathbf{X}_i^T\mathbf{x}}{b_1^2} \right) K\left(\left\| \frac{\mathbf{z}-\mathbf{Z}_i}{b_2} \right\|_2^2 \right),

where :math:`L(\cdot)` and :math:`K(\cdot)` are the directional and linear kernel functions while :math:`b_1,b_2` are the smoothing bandwidth parameters for directional and linear components, respectively. The challenge lies in the formulation of the *DirLinSCMS* algorithm, in that a naive generalization from the mean shift algorithm to its SCMS counterpart as how the standard SCMS and *DirSCMS* methods use will lead to a biased estimate of :math:`R_d(\widehat{f}_{dl})`; see Section 4 in [8]_. Fortunately, under the applications of the von Mises (directional) kernel and Gaussian (linear) kernel, we are able to formulate the correct SCMS iterative formula under the directional-linear data scenario as:

.. math::

    \begin{pmatrix}
	\mathbf{x}^{(t+1)}\\
	\mathbf{z}^{(t+1)}
	\end{pmatrix} \gets 
  \begin{pmatrix}
	\mathbf{x}^{(t)}\\
	\mathbf{z}^{(t)}
	\end{pmatrix}  +\eta \cdot \widehat{V}_{dl}\left(\mathbf{x}^{(t)},\mathbf{z}^{(t)}\right) \widehat{V}_{dl}\left(\mathbf{x}^{(t)},\mathbf{z}^{(t)}\right)^T \mathbf{H}\cdot \begin{pmatrix}
	\frac{\sum\limits_{i=1}^n \mathbf{X}_i\cdot L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b_1^2} \right)  K\left(\left\|\frac{\mathbf{z}^{(t)}-\mathbf{Z}_i}{b_2} \right\|_2^2 \right) }{\sum\limits_{i=1}^n L'\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b_1^2} \right) K\left(\left\|\frac{\mathbf{z}^{(t)}-\mathbf{Z}_i}{b_2} \right\|_2^2 \right)} -\mathbf{x}^{(t)}\\ \frac{\sum\limits_{i=1}^n \mathbf{Z}_i \cdot L\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b_1^2} \right)   K'\left(\left\|\frac{\mathbf{z}^{(t)}-\mathbf{Z}_i}{b_2} \right\|_2^2 \right) }{\sum\limits_{i=1}^n L\left(\frac{1-\mathbf{X}_i^T\mathbf{x}^{(t)}}{b_1^2} \right)  K'\left(\left\|\frac{\mathbf{z}^{(t)}-\mathbf{Z}_i}{b_2} \right\|_2^2 \right)} - \mathbf{z}^{(t)}
    \end{pmatrix},

.. math::

    \text{ and } \quad \mathbf{x}^{(t+1)} \gets \frac{\mathbf{x}^{(t+1)}}{\left\| \mathbf{x}^{(t+1)} \right\|_2}
    
for :math:`t=0,1,...`, where :math:`\widehat{V}_{dl}(\mathbf{x},\mathbf{z})=\left[\widehat{\mathbf{v}}_{d+1}(\mathbf{x},\mathbf{z}),..., \widehat{\mathbf{v}}_{q+D}(\mathbf{x},\mathbf{z})\right] \in \mathbb{R}^{(q+1+D)\times (q+D-d)}` has its columns as the last :math:`(q+D-d)` eigenvectors of the estimated Riemannian Hessian :math:`\mathcal{H} \widehat{f}_{dl}(\mathbf{x},\mathbf{z})` within the tangent space of :math:`\mathbb{S}^q\times \mathbb{R}^D` at :math:`(\mathbf{x},\mathbf{z})` and :math:`\mathbf{H}=\mathtt{Diag}`

    



References
----------

.. [1] Zhang, Y., de Souza, R. S., and Chen, Y.-C. (2022+). SCONCE: A Cosmic Web Finder for Spherical and Conic Geometries.
.. [2] Ozertem, U. and Erdogmus, D. (2011). Locally defined principal curves and surfaces. *Journal of Machine Learning Research*, **12**, 1249-1286.
.. [3] Genovese, C.R., Perone-Pacifico, M., Verdinelli, I. and Wasserman, L. (2014). Nonparametric ridge estimation. *The Annals of Statistics*, **42**(4), 1511-1545.
.. [4] Chen, Y.-C., Ho, S., Freeman, P.E., Genovese, C.R. and Wasserman, L. (2015). Cosmic web reconstruction through density ridges: method and algorithm. *Monthly Notices of the Royal Astronomical Society*, **454**(1), 1140-1156.
.. [5] Zhang, Y. and Chen, Y.-C. (2022). Linear convergence of the subspace constrained mean shift algorithm: from Euclidean to directional data. *Information and Inference: A Journal of the IMA*, iaac005, `https://doi.org/10.1093/imaiai/iaac005 <https://doi.org/10.1093/imaiai/iaac005>`_.
.. [6] Hall, P., Watson, G.S. and Cabrera, J. (1987). Kernel density estimation with spherical data. *Biometrika*, **74**(4), 751-762.
.. [7] García–Portugués, E. (2013). Exact risk improvement of bandwidth selectors for kernel density estimation with directional data. *Electronic Journal of Statistics*, **7**, 1655-1685.
.. [8] Zhang, Y. and Chen, Y.-C. (2021). Mode and ridge estimation in euclidean and directional product spaces: A mean shift approach. *arXiv preprint arXiv:2110.08505*, `https://arxiv.org/abs/2110.08505 <https://arxiv.org/abs/2110.08505>`_.
.. [9] García-Portugués, E., Crujeiras, R.M. and González-Manteiga, W. (2013). Kernel density estimation for directional–linear data. *Journal of Multivariate Analysis*, **121**, 152-175.
