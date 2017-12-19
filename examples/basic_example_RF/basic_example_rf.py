"""
======================================
How to fit a basic Monotone RF model
======================================

This example shows how to fit a model using `MonoRandomForestClassifier`.
In short, it is identical to `sci-kit learn's` RandomForestClassifier, with
the addition of `incr_feats`, `decr_feats` and `coef_calc_type` constructor
parameters, which describe the monotone features and the coefficient
calculation method. It fits a forest ensemble, converts the leaves into
monotone compliant 'rules', an then recalculates the rule coefficients
as described in the Theory section of this documentation.
"""

import numpy as np
from monoensemble import MonoRandomForestClassifier
from sklearn.datasets import load_boston

###############################################################################
# Load the data
# ----------------
#
# First we load the standard data source on 
# `Boston Housing 
# <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_, and 
# convert the output from real valued (regression) to binary classification 
# with roughly 50-50 class distribution:
#

data = load_boston()
y = data['target']
X = data['data']
features = data['feature_names']

###############################################################################
# Specify the monotone features
# -------------------------
# There are 13 predictors for house price in the Boston dataset:

###############################################################################
##. CRIM - per capita crime rate by town
##. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
##. INDUS - proportion of non-retail business acres per town.
##. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
##. NOX - nitric oxides concentration (parts per 10 million)
##. RM - average number of rooms per dwelling
##. AGE - proportion of owner-occupied units built prior to 1940
##. DIS - weighted distances to five Boston employment centres
##. RAD - index of accessibility to radial highways
##. TAX - full-value property-tax rate per $10,000
##. PTRATIO - pupil-teacher ratio by town
##. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
##. LSTAT - % lower status of the population
#
# The output is MEDV - Median value of owner-occupied homes in $1000's, but we 
# convert it to a binary y in +/-1 indicating whether MEDV is less than 
# $21(,000):

y[y< 21]=-1 # convert real output to 50-50 binary classification
y[y>=21]=+1 

###############################################################################
# We suspect that the number of rooms (6. RM) and the highway 
# accessibility (9. RAD) would, if anything, increase the price of a house
# (all other things being equal). Likewise we suspect that crime rate (1.
# CRIM), distance from employment (8. DIS) and percentage of lower status
# residents (13. LSTAT) would be likely to, if anything, decrease house prices.
# So we have:

incr_feats=[6,9]
decr_feats=[1,8,13]

###############################################################################
# Fit a MonoRandomForestClassifier model
# -------------------------
# We now fit our classifier. To understand the model and hyperparameters, 
# please refer to the original paper available 
# `here <http://staffhome.ecm.uwa.edu.au/~19514733/>`_:

# Specify hyperparams for model solution (normally optimised via oob_score or 
# CV)
n_estimators = 100
mtry = 3
coef_calc_type='boost' # or 'bayes','logistic'
# Solve model
clf = MonoRandomForestClassifier(n_estimators=n_estimators,
                                  max_features=mtry,
                                  oob_score=True,
                                  random_state=11,
                                  incr_feats=incr_feats,
                                  decr_feats=decr_feats,
                                  coef_calc_type=coef_calc_type)
clf.fit(X, y)

# Assess the model
y_pred = clf.predict(X)
acc_insample = np.sum(y == y_pred) / len(y)
mcr_oob=clf.oob_score_

###############################################################################
# Final notes
# -----------------------
# In a real scenario we would optimise the hyperparameter `mtry` using using
# the out-of-box (oob) score or cross-validation but this is 
# not covered in these basic examples. Enjoy!
