from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
from monoensemble import MonoGradientBoostingClassifier
from sklearn.datasets import load_boston


def load_data_set():
    # Load data
    max_N = 200
    data = load_boston()
    y = data['target']
    X = data['data']
    features = data['feature_names']
    multi_class = False
    # Specify monotone features
    incr_feat_names = ['RM', 'RAD']
    decr_feat_names = ['CRIM', 'DIS', 'LSTAT']
    # get 1 based indices of incr and decr feats
    incr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in incr_feat_names]
    decr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in decr_feat_names]
    # Convert to classification problem
    if multi_class:
        y_class = y.copy()
        thresh1 = 15
        thresh2 = 21
        thresh3 = 27
        y_class[y > thresh3] = 3
        y_class[np.logical_and(y > thresh2, y <= thresh3)] = 2
        y_class[np.logical_and(y > thresh1, y <= thresh2)] = 1
        y_class[y <= thresh1] = 0
    else:  # binary
        y_class = y.copy()
        thresh = 21  # middle=21
        y_class[y_class < thresh] = -1
        y_class[y_class >= thresh] = +1
    return X[0:max_N, :], y_class[0:max_N], incr_feats, decr_feats


# Load data
X, y, incr_feats, decr_feats = load_data_set()


def test_model_fit():
    # Specify hyperparams for model solution
    n_estimators = 1000
    subsample = 1.0
    learning_rate = 0.1
    max_depth = 3
    coef_calc_types = ['boost', 'bayes', 'logistic']
    insample_correct = [0.93999999999, 0.974999999, 0.9849999999]
    for i_test in np.arange(len(coef_calc_types)):
        coef_calc_type = coef_calc_types[i_test]
        # Solve model
        clf = MonoGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth,
            coef_calc_type=coef_calc_type, incr_feats=incr_feats,
            decr_feats=decr_feats, n_estimators=n_estimators,
            subsample=subsample, random_state=11)
        clf.fit(X, y)
        # Assess fit
        y_pred = clf.predict(X)
        acc = np.sum(y == y_pred) / len(y)
        npt.assert_almost_equal(0 if (acc - insample_correct[i_test]) <= 0.02
                                else 1, 0)


test_model_fit()
