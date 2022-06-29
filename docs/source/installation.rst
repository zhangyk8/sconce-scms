Installation guide
==================

Dependencies
------------

* Python >= 3.6 (earlier version might be applicable).
* `Numpy <http://www.numpy.org/>`_, `SciPy <https://www.scipy.org/>`_ (A speical function `scipy.special.iv <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv>`_ is used to compute the modified Bessel function of the first kind of real order).
* For the parallel implementation of the (extended) SCMS algorithms, ``sconce-scms`` leverages `Ray <https://ray.io/>`_ , a fast and simple distributed computing API for Python and Java.
* (Optional) To visualize the estimated filaments by ``sconce-scms``, one can resort to the `Matplotlib <https://matplotlib.org/>`_ package in python, especially the `Basemap <https://matplotlib.org/basemap/>`_ toolkit. We provide a `guideline <https://github.com/zhangyk8/DirMS/blob/main/Install_Basemap_Ubuntu.md>`_  about installing the `Basemap <https://matplotlib.org/basemap/>`_ toolkit on Ubuntu.


Quick Start
------------

To use a stable release of **SCONCE-SCMS**, install it using pip::

    pip install sconce-scms

The Pypi version is updated regularly. For the latest update, however, one should clone from GitHub and install it directly::

    git clone https://github.com/zhangyk8/sconce-scms.git
    cd sconce-scms
    python setup.py
