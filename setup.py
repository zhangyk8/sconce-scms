#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sconce-scms",
    version="0.1.1",
    author="Yikun Zhang",
    author_email="yikunzhang@foxmail.com",
    description="Spherical and Conic Cosmic Web Finders with Extended SCMS Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyk8/sconce-scms",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["sconce"]),
    install_requires=["numpy", "scipy", "ray[default]", "scikit-learn"],
    python_requires=">=3.6",
)
