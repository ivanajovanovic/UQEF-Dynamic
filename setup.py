#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='uqef_dynamic',
    version="0.1",
    url='https://github.com/ivanajovanovic/UQEF-Dynamic',
    author="Ivana Jovanovic Buha",
    author_email='ivana.jovanovic@tum.de',
    license='GNU GPL',
    platforms='any',
    packages=find_packages("uqef_dynamic"),
    package_dir={"": "uqef_dynamic"},
    install_requires=[
        'chaospy',
        'uqef',
        'dill',
        'joblib',
        'matplotlib',
        'more_itertools',
        'mpi4py',
        'numpy',
        'pandas',
        'plotly',
        'pyproj',
        'scikit-learn',
        'scipy',
        'seaborn',
        'setuptools'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU GPL-3.0 License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
)
