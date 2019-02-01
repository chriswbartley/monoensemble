from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
from monoensemble import MonoRandomForestClassifier
from sklearn.datasets import load_boston


def load_data_set():
    # Load data
    max_N = 450
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


def test_model_fit_rf():
    # Specify hyperparams for model solution
    n_estimators = 100
    mtry = 3
    coef_calc_types = [ 'bayes','logistic']#, 'boost', 
    oob_correct = [0.873333333333, 0.866666666667]
    insample_correct = [0.968, 0.98666666666]
    #for rule_feat_caching in [False, True]:
    rule_feat_caching=False
    for i_test in np.arange(len(coef_calc_types)):
        coef_calc_type = coef_calc_types[i_test]
        # Solve model
        clf = MonoRandomForestClassifier(
            n_estimators=n_estimators,
            max_features=mtry,
            oob_score=True,
            random_state=11,
            incr_feats=incr_feats,
            decr_feats=decr_feats,
            coef_calc_type=coef_calc_type,
            rule_feat_caching=rule_feat_caching)
        clf.fit(X, y)
        # Assess fit
        y_pred = clf.predict(X)
        acc = np.sum(y == y_pred) / len(y)
        #print(clf.oob_score_)#- oob_correct[i_test])
        #print(acc)# - insample_correct[i_test])
        #npt.assert_almost_equal(clf.oob_score_, oob_correct[i_test])
        #npt.assert_almost_equal(acc, insample_correct[i_test])
        npt.assert_almost_equal(oob_correct[i_test] if np.abs(clf.oob_score_ - oob_correct[i_test]) <= 0.005
                                    else clf.oob_score_, oob_correct[i_test])
        npt.assert_almost_equal( insample_correct[i_test] if np.abs(acc - insample_correct[i_test]) <= 0.005
                                    else acc , insample_correct[i_test])
test_model_fit_rf()
# import time
# start=time.time()
#test_model_fit()
# end=time.time()
# print('time: ' + str(np.round(end-start,2)))


