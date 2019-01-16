from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
from monoensemble import MonoRandomForestClassifierFSD
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
    incr_feats=np.asarray(incr_feats)
    decr_feats=np.asarray(decr_feats)
    mt_feat_types = np.zeros(len(features),dtype=np.int32)
    if len(incr_feats)>0:
        mt_feat_types [incr_feats-1]=1
    if len(decr_feats)>0:
        mt_feat_types [decr_feats-1]=-1
    # Convert to classification problem
    # Multi-class
    y_multiclass = y.copy()
    thresh1 = 15
    thresh2 = 21
    thresh3 = 27
    y_multiclass[y > thresh3] = 3
    y_multiclass[np.logical_and(y > thresh2, y <= thresh3)] = 2
    y_multiclass[np.logical_and(y > thresh1, y <= thresh2)] = 1
    y_multiclass[y <= thresh1] = 0
    # Binary
    y_binary = y.copy()
    thresh = 21  # middle=21
    y_binary[y_binary < thresh] = -1
    y_binary[y_binary >= thresh] = +1
    return X, y_binary, y_multiclass, incr_feats, decr_feats, mt_feat_types


# Load data
max_N = 200#200
np.random.seed(13) # comment out for changing random training set
X, y_binary, y_multiclass, incr_feats, decr_feats, mt_feat_types = load_data_set()
indx_train=np.random.permutation(np.arange(X.shape[0]))[0:max_N]
inx_test=np.asarray([i for i in np.arange(max_N) if i not in indx_train ])
X_train=X[indx_train,:]
X_test=X[inx_test,:]

def test_model_fit():
    # Specify hyperparams for model solution
    n_estimators = 300
    mtry = 3
    #coef_calc_types = ['boost', 'bayes', 'logistic']
    oob_correct = [0.8649999999999, 0.85999999999, 0.86499999999999]
    insample_correct = [0.949999999999, 0.9649999999999, 0.98499999999]
    #for rule_feat_caching in [False, True]:
        #for i_test in np.arange(len(coef_calc_types)):
            #coef_calc_type = coef_calc_types[i_test]
    # Solve model
    for y in [y_binary,y_multiclass]:
        for mt_feat_types_ in [None,mt_feat_types]:
            y_train=y[indx_train]
            y_test=y[inx_test]
            clf = MonoRandomForestClassifierFSD(
                n_estimators=n_estimators,
                max_features=mtry,
                oob_score=True,
                random_state=11,
                mt_feat_types=mt_feat_types_)
            clf.fit(X_train, y_train)
            # Assess fit
            y_pred = clf.predict(X_test)
            acc = np.sum(y_test == y_pred) / len(y_test)
        # print(clf.oob_score_- oob_correct[i_test])
        # print(acc - insample_correct[i_test])
            #print(clf.oob_score_)
            print(acc)
    #npt.assert_almost_equal(clf.oob_score_, oob_correct[i_test])
    #npt.assert_almost_equal(acc, insample_correct[i_test])
        
# import time
# start=time.time()
test_model_fit()
# end=time.time()
# print('time: ' + str(np.round(end-start,2)))


