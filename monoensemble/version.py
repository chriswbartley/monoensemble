from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 1
_version_minor = 3
_version_micro = 3  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: BSD 3 Clause License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = """
monoensemble: a high performance library for monotone Gradient Boosting and Random Forest Classification.
"""
# Long description will go up on the pypi page
long_description = """

monoensemble
========
This package contains high performance implementations of 
MonoGradientBoostingClassifier and MonoRandomForestClassifier. These monotone
classifiers use the fast and scaleable monotone rule approach described
in Bartley C., Liu W., and Reynolds M. \Enhanced Random Forest Algorithms for
Partially Monotone Ordinal Classication". In: Proceedings of the Thirty-Third AAAI Conference on
Articial Intelligence (AAAI-2019), Honolulu, Hawaii, USA, Jan 27 - Feb 1 PREPRESS. ed. by
AAAI Press. AAAI. 2019.

To get started, please go to the repository README_.

.. _README: https://github.com/chriswbartley/monoensemble/blob/master/README.md


License
=======
``monoensemble`` is licensed under the terms of the BSD 3 Clause License. See the
file "LICENSE" for information on the history of this software, terms &
conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017, Christopher Bartley
"""

NAME = 'monoensemble'
MAINTAINER = "Christopher Bartley"
MAINTAINER_EMAIL = "christopher.bartley@research.uwa.edu.au"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/chriswbartley/monoensemble"
DOWNLOAD_URL = ""
LICENSE = "BSD 3 Clause"
AUTHOR = "Christopher Bartley"
AUTHOR_EMAIL = "christopher.bartley@research.uwa.edu.au"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'monoensemble': [pjoin('data', '*')]}
REQUIRES = []
INSTALL_REQUIRES = ["numpy","scipy","scikit-learn", "cython"]
