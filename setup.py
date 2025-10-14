#!/usr/bin/env python
"""
Setup script for zero-shot coreset selection FiftyOne plugin.
"""

from setuptools import setup, find_packages

setup(
    name="fiftyone-zero-shot-coreset-selection",
    version="1.0.0",
    description="Zero-shot coreset selection plugin for FiftyOne",
    author="Voxel51",
    url="https://github.com/voxel51/zero-shot-corest-selection",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "fiftyone>=0.23.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
