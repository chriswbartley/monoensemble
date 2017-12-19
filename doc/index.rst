.. monoensemble documentation master file, created by sphinx-quickstart on Tue Apr 14 10:29:06 2015. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. role:: bash(code)
   :language: bash

Welcome to monoensemble's documentation!
====================================

This package contains two key classification algorithms:
 - `MonoRandomForestClassifier` is a Random Forest classifier with the added capability of partially monotone features. It is very fast and demonstrates excellent experimental accuracy. 
 - `MonoGradientBoostingClassifier` is a monotone Gradient Boosting classifier. It is very fast and demonstrates very good experimental accuracy.

These algorithms are heavily based on (or inherit from) sci-kit learn's versions, and the interface is identical except that the constructor has three additional parameters:
 - incr_feats : The one-based array indices of the columns in X that should only have a monotone increasing impact on the resulting class.
 - decr_feats : The one-based array indices of the columns in X that should only have a monotone decreasing impact on the resulting class.
 - coef_calc_type : string
        Determines how the rule coefficients are calculated. Allowable values:
'boost' DEFAULT: A single Newton step approximation is used. Fast, and generally best. 
'bayesian': Assumes conditional indpendence between rules and calculates coefficients as per Naive bayesian classification. Fast with good results.
'logistic': L2 regularised logistic regression. Slower.

To install, simply use :bash:`pip install monoensemble`. For full documentation you've come to the right place. For a brief overview, refer to the `README file 
<https://github.com/chriswbartley/monoensemble/blob/master/README.md>`_ in the Github repository.

Contents:

.. toctree::
   :maxdepth: 2

   theory
   auto_examples/index
   api
