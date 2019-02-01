# -*- coding: utf-8 -*-
"""Monotone Gradient Boosted Trees

This module contains methods for fitting gradient boosted trees for
classification. Monotonicity in the requested features is achieved using the
technique from Bartley C., Liu W., and Reynolds M. 2017, ``Fast & Perfect
Monotone Random Forest Classification``, prepub, PAKDD submission, available
here_(http://staffhome.ecm.uwa.edu.au/~19514733/). Multi-class classification
is implemented using the monotone ensembling procedure from Kotlowski W, and
Slowinski R., 2013 ``On nonparametric ordinal classification with monotonicity
constraints``, IEEE Transactions on Knowledge and Data Engineering, vol 25,
no. 11, 2576--2589.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used. At present only
  classification is implemented, with the Binomial Deviance loss function.

- ``MonoGradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` NOT IMPLEMENTED YET (see commented out code).
"""
# Author: Christopher Bartley
# Based on, and heavily reusing, original sci-kit learn gradient_boosting, by:
#          Peter Prettenhofer, Scott White, Gilles Louppe, Emanuele Olivetti,
#          Arnaud Joly, Jacob Schreiber
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin
# from sklearn.base import RegressorMixin
from sklearn.externals import six


from monoensemble import apply_rules_c
from monoensemble import get_node_map_c
from monoensemble import update_rule_coefs
from monoensemble import update_rule_coefs_newton_step
from monoensemble import _random_sample_mask
from monoensemble import get_node_map_and_rule_feats_c
from monoensemble import apply_rules_rule_feat_cache_c
from monoensemble import apply_rules_set_based_c
from monoensemble import extract_rules_from_tree_c
from monoensemble import apply_rules_from_tree_sorted_c
from monoensemble import _log_logistic_sigmoid
from monoensemble import  _custom_dot
from monoensemble import  _custom_dot_multiply
from monoensemble import calc_newton_step_c
import numbers
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import issparse
from scipy.special import expit

from time import time
from sklearn.tree.tree import DecisionTreeRegressor as DecisionTreeRegressorSklearn

from sklearn.tree._tree import DTYPE
from sklearn.tree._tree import TREE_LEAF

from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.utils import deprecated
# from sklearn.utils.fixes import logsumexp
# from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError

# For Constrained Logistic Regression:
import scipy as sp
from scipy import optimize as opt
from sklearn.utils.extmath import (log_logistic, safe_sparse_dot)

RULE_LOWER_CONST = -1e9
RULE_UPPER_CONST = 1e9

MIN_NODE_SIZE_FOR_SORTING_=5
class LogOddsEstimator(object):
    """An estimator predicting the log odds ratio."""
    scale = 1.0

    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def fit(self, X, y, sample_weight=None):
        lidstone_alpha = 0.01
        if self.n_classes <= 2:
            # pre-cond: pos, neg are encoded as 1, 0
            if sample_weight is None:
                pos = np.sum(y)
                neg = y.shape[0] - pos
            else:
                pos = np.sum(sample_weight * y)
                neg = np.sum(sample_weight * (1 - y))

            if neg == 0 or pos == 0:
                raise ValueError('y contains non binary labels.')
            self.prior = self.scale * np.log(pos / neg)
        else:  # multi-class ordinal
            self.prior = np.zeros(self.n_classes - 1, dtype=np.float64)
            for k in np.arange(self.n_classes - 1):
                if sample_weight is None:
                    pos = np.sum(y > k)
                    neg = y.shape[0] - pos
                else:
                    pos = np.sum(sample_weight * (y > k).astype(np.float64))
                    neg = np.sum(sample_weight *
                                 (1 - (y > k).astype(np.float64)))
                if pos == 0 or neg == 0:
                    pos = pos + lidstone_alpha
                    neg = neg + 2 * lidstone_alpha
                self.prior[k] = self.scale * np.log(pos / neg)

    def predict(self, X):
        check_is_fitted(self, 'prior')
        if np.isscalar(self.prior):
            y = np.empty((X.shape[0], 1), dtype=np.float64)
            y.fill(self.prior)
        else:  # multi-class
            y = np.tile(self.prior, [X.shape[0], 1]).astype(np.float64)
        return y


class ScaledLogOddsEstimator(LogOddsEstimator):
    """Log odds ratio scaled by 0.5 -- for exponential loss. """
    scale = 0.5


class PriorProbabilityEstimator(object):
    """An estimator predicting the probability of each
    class in the training data.
    """

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        class_counts = np.bincount(y, weights=sample_weight)
        self.priors = class_counts / class_counts.sum()

    def predict(self, X):
        check_is_fitted(self, 'priors')

        y = np.empty((X.shape[0], self.priors.shape[0]), dtype=np.float64)
        y[:] = self.priors
        return y


class ZeroEstimator(object):
    """An estimator that simply predicts zero. """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes
    def fit(self, X, y, sample_weight=None):
        if np.issubdtype(y.dtype, np.signedinteger):
            # classification
            if self.n_classes is None:
                self.n_classes = np.unique(y).shape[0]
            if self.n_classes == 2:
                self.n_classes = 1
            else:
                self.n_classes = self.n_classes
        else:
            # regression
            self.n_classes = 1

    def predict(self, X):
        check_is_fitted(self, 'n_classes')

        y = np.empty((X.shape[0], 1 if self.n_classes <=
                      2 else self.n_classes - 1), dtype=np.float64)
        y.fill(0.0)
        return y


class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    incr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone increasing impact on the resulting class.
    decr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone decreasing impact on the resulting class.
    coef_calc_type : string
        Determines how the rule coefficients are calculated. Allowable values:
        'logistic': L2 regularised logistic regression. Slower.
        'boost': A single Newton step approximation is used. Fast, and
        generally best.
        'bayesian': Assumes conditional indpendence between rules and
        calculates coefficients as per Naive bayesian classification. Fast
        with good results.

    """

    is_multi_class = False

    def __init__(
            self,
            K,
            incr_feats=[],
            decr_feats=[],
            coef_calc_type='boost'):
        self.K = K
        self.incr_feats = np.asarray(incr_feats)
        self.decr_feats = np.asarray(decr_feats)
        self.coef_calc_type = 0 if coef_calc_type == 'boost' else 3 if (
            coef_calc_type == 'logistic') else 2  # bayes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.

        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

        
    @abstractmethod
    def update_terminal_rules(
            self,
            rule_mask,
            rule_values,
            y,
            residual,
            y_pred,
            sample_weight,
            learning_rate=1.0,
            k=0,
            logistic_intercept=True):
        """Update the terminal rules (each derived from a leaf)
        and update the current predictions of the model.

        Parameters
        ----------
        rule_lower_corners : ndarray, shape=(L,m) where R is number of rules
            Lower corners of the rule region  hypercubes, 1 per rule.
        rule_upper_corners : ndarray, shape=(L,m) where R is number of rules
            Upper corners of the rule region  hypercubes, 1 per rule.
        rule_values : ndarray, shape = (L,)
            original rule values (i.e. the original tree leaf node values)
        X : ndarray, shape=(n, m)
            The data array.
        y : ndarray, shape=(n,)
            The target labels.
        residual : ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : ndarray, shape=(n,)
            The predictions.
        sample_weight : ndarray, shape=(n,)
            The weight of each sample.
        sample_mask : ndarray, shape=(n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.
        X_leaf_node_ids : ndarray, shape=(n,)
            The index of the leaf node for each training point in X
        node_rule_map : ndarray, shape=(T,R)
            Each row t is a list of the rule indices that node t of tree
            overlaps. The list starts at column 0 and proceeds to each adjacent
            column until there are no more overlapping rules, where it has
            value -99 for all remaining columns. Speeds up rule assessment.
        logistic_intercept : boolean, default True
            True if the logistic regression should fit an intercept. Only
            relevant if self.coef_calc_type==3 (logistic).
        """

    @abstractmethod
    def _update_terminal_rule(self, rule_lower_corners, rule_upper_corners,
                              rule_values, rule_mask,
                              i_rule, X, y, residual, pred, sample_weight):
        """Template method for updating terminal rules (=leaves). """


class ClassificationLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for classification loss functions. """

    def _score_to_proba(self, score):
        """Template method to convert scores to probabilities.

         the does not support probabilities raises AttributeError.
        """
        raise TypeError(
            '%s does not support predict_proba' %
            type(self).__name__)

    @abstractmethod
    def _score_to_decision(self, score):
        """Template method to convert scores to decisions.

        Returns int arrays.
        """


class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    """

    def __init__(
            self,
            n_classes,
            coef_calc_type='boost',
            incr_feats=[],
            decr_feats=[]):
        # we only need to fit one tree for binary clf.
        self.is_multi_class = True if n_classes > 2 else False
        super(
            BinomialDeviance,
            self).__init__(
            1 if n_classes == 2 else n_classes,
            coef_calc_type=coef_calc_type,
            incr_feats=incr_feats,
            decr_feats=decr_feats)

    def init_estimator(self):
        return LogOddsEstimator(n_classes=2 if self.K <= 2 else self.K)

    def __call__(self, y, pred, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood). """
        if self.K <= 2:
            pred = pred.ravel()
            if sample_weight is None:
                return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))
            else:
                return (-2.0 / sample_weight.sum() *
                        np.sum(sample_weight * ((y * pred) -
                                                np.logaddexp(0.0, pred))))
        else:
            # create ordinal encoding
            Y = np.zeros((y.shape[0], self.K - 1), dtype=np.float64)
            loss__ = np.zeros(self.K - 1)
            for k in range(self.K - 1):
                Y[:, k] = y > k
                if sample_weight is None:
                    loss__[k] = -2.0 * np.mean((Y[:, k] * pred[:, k]) -
                                               np.logaddexp(0.0, pred[:, k]))
                else:
                    loss__[k] = (-2.0 / sample_weight.sum() *
                                 np.sum(sample_weight * (
                                        (Y[:, k] * pred[:, k]) -
                                        np.logaddexp(0.0, pred[:, k]))))
            return np.sum(loss__)

    def negative_gradient(self, y, pred, k=0, **kargs):
        """Compute the residual (= negative gradient). """
        return y - expit(pred[:, k].ravel())

    #  NO LONGER USED, USES CYTHON VERSION FOR SPEED  #
    def _update_terminal_rule(self, leaf_mask,
                              rule_values, rule_mask,
                              i_rule, X, y, residual, pred, sample_weight):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual"""
        this_rule_mask = rule_mask[:, i_rule]
        this_rule_val = rule_values[i_rule]
        residual_ = residual[this_rule_mask]
        pred_ = pred[this_rule_mask]
        y_ = y[this_rule_mask]
        sample_weight_ = sample_weight[this_rule_mask]
        if self.coef_calc_type == 0 or self.coef_calc_type == 1:  # 'boost':
            numerator = np.sum(sample_weight_ * residual_)
            denominator = np.sum(
                sample_weight_ * (y_ - residual_) * (1 - y_ + residual_))
            # prevents overflow and division by zero
            if self.coef_calc_type == 0:
                if abs(denominator) < 1e-150:
                    coef_ = 0.0
                else:
                    coef_ = numerator / denominator
            elif self.coef_calc_type == 1:  # use leaf coef if rule dirn is ok
                coef_ = 0.0
                if np.sign(numerator) * np.sign(denominator) * \
                        np.sign(this_rule_val) >= 0:
                    this_leaf_mask = leaf_mask[:, i_rule]
                    residual_ = residual[this_leaf_mask]
                    pred_ = pred[this_leaf_mask]
                    y_ = y[this_leaf_mask]
                    sample_weight_ = sample_weight[this_leaf_mask]
                    numerator = np.sum(sample_weight_ * residual_)
                    denominator = np.sum(sample_weight_ *
                                         (y_ - residual_) *
                                         (1 - y_ + residual_))
                    # prevents overflow and division by zero
                    if abs(denominator) < 1e-150:
                        coef_ = 0.0
                    else:
                        coef_ = numerator / denominator
        else:  # sel f.coef_calc_type=='bayes':
            lidstone_alpha = 0.01
            numerator = np.sum(y_ * sample_weight_) + lidstone_alpha
            denominator = np.sum(sample_weight_) + lidstone_alpha * 2.
            prob1 = numerator / denominator
            prob1_pred = (np.sum(expit(pred_) * sample_weight_) +
                          lidstone_alpha) / denominator
            coef_ = np.log(prob1 / (1 - prob1)) - \
                np.log(prob1_pred / (1 - prob1_pred))
        # apply constraints
        if np.sign(this_rule_val) * np.sign(coef_) >= 0:
            return coef_
        else:
            return 0.0

    def update_terminal_rules(
            self,
            rule_mask,
            rule_values,
            y,
            residual,
            y_pred,
            sample_weight,
            learning_rate=1.0,
            k=0,
            logistic_intercept=True,
            adjust_intercept_=False):

        intercept_ = 0.
        if self.coef_calc_type == 3:  # 'logistic'
            sample_mask = sample_weight > 0
            y_ = sp.float64(y[sample_mask].copy())
            sample_weight_ = sp.float64(
                sample_weight[sample_mask].astype(sp.float64).reshape([-1, 1]))
            rule_mask_ = sp.float64(rule_mask[sample_mask, :])
            pred_ = sp.float64(y_pred[sample_mask, k])
            offset_ = sp.float64(
                np.hstack([-pred_.reshape([-1, 1]), pred_.reshape([-1, 1])]))
            x_ = sp.float64(rule_mask_)
            # default, no limits on coefs
            coef_limits = sp.array(
                [[sp.float64(-sp.inf)], [sp.float64(sp.inf)]])
            if len(self.incr_feats) > 0 or len(self.decr_feats) > 0:
                coef_limits = np.zeros([2, rule_mask_.shape[1]])
                for i_r in np.arange(rule_mask_.shape[1]):
                    coef_limits[0, i_r] = - \
                        np.inf if rule_values[i_r] <= 0 else 0.
                    coef_limits[1,
                                i_r] = np.inf if rule_values[i_r] >= 0 else 0.
                coef_limits = sp.array(coef_limits)
            # simplify to unique rows
            all_yx = np.hstack([y_.reshape([-1, 1]), offset_, x_])
            all_yx_uniq, inverse_idx = np.unique(
                all_yx, return_inverse=True, axis=0)
            y_uniq = sp.float64(all_yx_uniq[:, 0].reshape([-1, 1]))
            offset_uniq = sp.float64(all_yx_uniq[:, 1:3].reshape([-1, 2]))
            x_uniq = sp.float64(all_yx_uniq[:, 3:])
            if np.isscalar(y_uniq):  # catch special case where all y=0 or y=1
                if logistic_intercept:  # need to add an intercept
                    # use naive probability estimate, need lidstone smoothing
                    # to avoid infinite results
                    lidstone_alpha = 0.01
                    n = np.sum(sample_weight[y == y_uniq])
                    p_ = (lidstone_alpha + y_uniq * n) / \
                        (n + 2 * lidstone_alpha)
                    intercept_ = np.log(p_ / (1 - p_))
                else:
                    intercept_ = 0.
            else:  # have a mix of y worth solving!
                sample_weight_uniq = sp.float64(np.zeros([len(y_uniq), 1]))
                for i_ in np.arange(all_yx_uniq.shape[0]):
                    sample_weight_uniq[i_, 0] = np.sum(
                        sample_weight_[inverse_idx == i_])
                C_ = 1e4
                standardize = False
                intercept = logistic_intercept
                incr_feats = np.arange(len(rule_values))[rule_values > 0] + 1
                decr_feats = np.arange(len(rule_values))[rule_values < 0] + 1
                logistic_conjug = ConstrainedLogisticRegression(
                    C=C_,
                    solver='newton-cg',
                    incr_feats=incr_feats,
                    decr_feats=decr_feats,
                    regularise_intercept=False,
                    standardize=standardize,
                    penalty='l2',
                    fit_intercept=intercept)
                logistic_conjug.fit(x_uniq.copy(), y_uniq.copy(
                ), sample_weight_uniq, offset=offset_uniq[:, 1])
                intercept_ = logistic_conjug.intercept_[0]
                coef_ = logistic_conjug.coef_[0]
                rule_values[:] = coef_
            #y_pred[:, k] += learning_rate * intercept_
        elif self.coef_calc_type == 2:  # 'bayes'
            
            lidstone_alpha = 0.01
            coefs = np.zeros(rule_mask.shape[1], dtype=np.float64)
            update_rule_coefs(rule_mask.astype(np.int32),
                              y_pred[:, k].astype(np.float64),
                              y.astype(np.int32),
                              sample_weight.astype(np.float64),
                              lidstone_alpha, coefs)
            rule_values[:] = np.where(rule_values * coefs >= 0, coefs, 0)
            if adjust_intercept_: # for bayes x 1 estimator optimisation
                intercept_=y_pred[0, k]
        elif self.coef_calc_type == 0:  # 'boost'
            coefs = np.zeros(rule_mask.shape[1], dtype=np.float64)
            update_rule_coefs_newton_step(
                rule_mask.astype(
                    np.int32), residual.astype(
                    np.float64), y.astype(
                    np.int32), sample_weight.astype(
                    np.float64), coefs)
            rule_values[:] = np.where(rule_values * coefs >= 0, coefs, 0)
            
# NO LONGER USED, USE CYTHON VERSION FOR SPEED
#        else:
#            # update rule coefs
#            for i_rule in np.arange(rule_mask.shape[1]):
#                rule_values[i_rule]=self._update_terminal_rule(leaf_mask,
#                                       rule_values,rule_mask,i_rule,X,y,
#                                       residual,
#                                       y_pred[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        
        for i_rule in np.arange(rule_mask.shape[1]):
            y_pred[:, k] += (learning_rate *
                             rule_mask[:, i_rule].astype(float) *
                             rule_values[i_rule])

        if adjust_intercept_: # for bayes approx optimisation
            y_pred[:, k] -= learning_rate * intercept_ # subject old intercept
            intercept_=adjust_intercept(intercept_,                                    
                      y_pred[:, k].astype(np.float64),
                      y.astype(np.int32),
                      sample_weight.astype(np.float64),
                      coefs)    
        y_pred[:, k] += learning_rate * intercept_
        return intercept_

    def _score_to_proba(self, score):
        if not self.is_multi_class:  # binary
            proba = np.ones((score.shape[0], 2), dtype=np.float64)
            proba[:, 1] = expit(score.ravel())
            proba[:, 0] -= proba[:, 1]
        else:
            proba = expit(score)
        return proba

    def _score_to_decision(self, score):
        if self.K <= 2:
            proba = self._score_to_proba(score)
            return np.argmax(proba, axis=1)
        else:
            proba = self._score_to_proba(score)
            return np.sum((proba > 0.5).astype(np.int), axis=1)

# TO BE IMPLEMENTED
# class ExponentialLoss(ClassificationLossFunction):
#    """Exponential loss function for binary classification.
#
#    Same loss as AdaBoost.
#
#    References
#    ----------
#    Greg Ridgeway, Generalized Boosted Models: A guide to the gbm package,2007
#    """
#    def __init__(self, n_classes):
#        if n_classes != 2:
#            raise ValueError("{0:s} requires 2 classes.".format(
#                self.__class__.__name__))
#        # we only need to fit one tree for binary clf.
#        super(ExponentialLoss, self).__init__(1)
#
#    def init_estimator(self):
#        return ScaledLogOddsEstimator()
#
#    def __call__(self, y, pred, sample_weight=None):
#        pred = pred.ravel()
#        if sample_weight is None:
#            return np.mean(np.exp(-(2. * y - 1.) * pred))
#        else:
#            return (1.0 / sample_weight.sum() *
#                    np.sum(sample_weight * np.exp(-(2 * y - 1) * pred)))
#
#    def negative_gradient(self, y, pred, **kargs):
#        y_ = -(2. * y - 1.)
#        return y_ * np.exp(y_ * pred.ravel())
#
#    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
#                                residual, pred, sample_weight):
#        terminal_region = np.where(terminal_regions == leaf)[0]
#        pred = pred.take(terminal_region, axis=0)
#        y = y.take(terminal_region, axis=0)
#        sample_weight = sample_weight.take(terminal_region, axis=0)
#
#        y_ = 2. * y - 1.
#
#        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * pred))
#        denominator = np.sum(sample_weight * np.exp(-y_ * pred))
#
#        # prevents overflow and division by zero
#        if abs(denominator) < 1e-150:
#            tree.value[leaf, 0, 0] = 0.0
#        else:
#            tree.value[leaf, 0, 0] = numerator / denominator
#
#    def _score_to_proba(self, score):
#        proba = np.ones((score.shape[0], 2), dtype=np.float64)
#        proba[:, 1] = expit(2.0 * score.ravel())
#        proba[:, 0] -= proba[:, 1]
#        return proba
#
#    def _score_to_decision(self, score):
#        return (score.ravel() >= 0.0).astype(np.int)
#
#
# LOSS_FUNCTIONS = {'ls': LeastSquaresError,
#                  'lad': LeastAbsoluteError,
#                  'huber': HuberLossFunction,
#                  'quantile': QuantileLossFunction,
#                  'deviance': None,    # for both, multinomial and binomial
#                  'exponential': ExponentialLoss,
#                  }


LOSS_FUNCTIONS = {'deviance': BinomialDeviance}

INIT_ESTIMATORS = {'zero': ZeroEstimator}


class VerboseReporter(object):
    """Reports verbose output to stdout.

    If ``verbose==1`` output is printed once in a while (when iteration mod
    verbose_mod is zero).; if larger than 1 then output is printed for
    each update.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        print(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration. """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10

def apply_rules_set_based(X,
        rule_lower_corners,
        rule_upper_corners):                

    # create output matrix
    rule_mask = np.zeros(
        [X.shape[0], rule_lower_corners.shape[0]], dtype=np.int32,order='C')
    sorted_feats=np.zeros(X.shape,dtype=np.float64,order='F')
    sorted_indxs=np.zeros(X.shape,dtype=np.int32,order='F')
    sorted_datapoint_posns=np.zeros(X.shape,dtype=np.int32,order='F')
    for j in np.arange(X.shape[1]):
        sorted_indxs[:,j]= np.argsort(X[:,j], axis=-1, kind='quicksort')
        sorted_feats[:,j]=X[sorted_indxs[:,j],j].copy()
    apply_rules_set_based_c(
        X.astype(
            np.float64),
        sorted_feats,
        sorted_indxs,
        sorted_datapoint_posns,
        rule_lower_corners,
        rule_upper_corners,
        rule_mask)  
    return np.asarray(rule_mask, dtype=bool)
        
# set banaive approachon, about 3X slower than cython naive approach
def apply_rules_sets_sorted(X,
        rule_lower_corners,
        rule_upper_corners,
        X_leaf_node_ids=None,
        node_rule_map=None):
    # sort X feats
    sorted_feats=np.zeros(X.shape,dtype=np.float64)
    sorted_indxs=np.zeros(X.shape,dtype=np.int32)
    sorted_datapoint_posns=np.zeros(X.shape,dtype=np.int32)
    for j in np.arange(X.shape[1]):
        sorted_indxs[:,j]= np.argsort(X[:,j], axis=-1, kind='quicksort')
        sorted_feats[:,j]=X[sorted_indxs[:,j],j]
        i=0
        for k in sorted_indxs[:,j]:
            sorted_datapoint_posns[k,j]=i
            i=i+1
    
    # apply each rule
    n=X.shape[0]
    rule_tx=np.zeros([X.shape[0],rule_lower_corners.shape[0]],dtype=np.int32)
    for r in np.arange(rule_lower_corners.shape[0]):
        feat_sets=np.zeros([X.shape[1]*2,4],dtype=np.int32)
        i_f=0
        for j in np.arange(X.shape[1],dtype=np.int32):
            if rule_lower_corners[r,j]!=RULE_LOWER_CONST:
                insert_pos=np.searchsorted(sorted_feats[:,j], 
                                           rule_lower_corners[r,j], 
                                           side='right')
                feat_sets[i_f,:]=[j,-1,insert_pos,n-insert_pos]
                i_f=i_f+1
            if rule_upper_corners[r,j]!=RULE_UPPER_CONST:
                insert_pos=np.searchsorted(sorted_feats[:,j], 
                                           rule_upper_corners[r,j], 
                                           side='right')
                feat_sets[i_f,:]=[j,1,insert_pos,insert_pos]
                i_f=i_f+1
        if i_f==0:
            viable_pts=np.arange(n,dtype=np.int32)
        else:
            feat_sets=feat_sets[0:i_f,:]
            feat_sets=feat_sets[feat_sets[:,3].argsort(),:]
            j=feat_sets[0,0]
            insert_pos=int(feat_sets[0,2])
            dirn=feat_sets[0,1]
            if dirn==-1:
                viable_pts=sorted_indxs[insert_pos:,j]
            else:
                viable_pts=sorted_indxs[0:insert_pos,j]
                
            for i_ff in np.arange(1,i_f):
                j=feat_sets[i_ff,0]
                dirn=feat_sets[i_ff,1]
                insert_pos=feat_sets[i_ff,2]
                if dirn==-1:
                    viable_pts=viable_pts[sorted_datapoint_posns[viable_pts,j]>=insert_pos]
                else:
                    viable_pts=viable_pts[sorted_datapoint_posns[viable_pts,j]<insert_pos]
        if len(viable_pts)>0:
            rule_tx[viable_pts,r] = 1
    return rule_tx
    
# set banaive approachon, about 3X slower than cython naive approach
def apply_rules_sets(X,
        rule_lower_corners,
        rule_upper_corners,
        X_leaf_node_ids=None,
        node_rule_map=None):
    # sort X feats
    sorted_feats=dict()
    sorted_indxs=dict()
    sorted_datapoint_posns=dict()
    for j in np.arange(X.shape[1]):
        sorted_indxs[j]= np.argsort(X[:,j], axis=-1, kind='quicksort')
        sorted_feats[j]=X[:,j][sorted_indxs[j]]
        posns=np.zeros(X.shape[0])
        i=0
        for k in sorted_indxs[j]:
            posns[k]=i
            i=i+1
        sorted_datapoint_posns[j]=posns
    # apply each rule
    rule_tx=np.zeros([X.shape[0],rule_lower_corners.shape[0]],dtype=np.int32)
    for r in np.arange(rule_lower_corners.shape[0]):
        viable_pts=np.arange(X.shape[0])
        for j in np.arange(X.shape[1]):
            if rule_lower_corners[r,j]!=RULE_LOWER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], 
                                           rule_lower_corners[r,j], side='right')
                #viable_pts_dict[j]=set(sorted_indxs[j][insert_pos:])
                viable_pts_this_feat=sorted_indxs[j][insert_pos:]
                viable_pts=viable_pts[sorted_datapoint_posns[j][viable_pts]>=insert_pos]  
            if rule_upper_corners[r,j]!=RULE_UPPER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], rule_upper_corners[r,j], side='right')
                viable_pts_this_feat=sorted_indxs[j][0:insert_pos]
                viable_pts=viable_pts[sorted_datapoint_posns[j][viable_pts]<insert_pos]
        if len(viable_pts)>0:
            rule_tx[viable_pts,r] = 1
    return rule_tx
def apply_rules_sets_hybrid(X,
        rule_lower_corners,
        rule_upper_corners,
        X_leaf_node_ids=None,
        node_rule_map=None):
    # sort X feats
    sorted_feats=dict()
    sorted_indxs=dict()
    for j in np.arange(X.shape[1]):
        sorted_indxs[j]= np.argsort(X[:,j], axis=-1, kind='quicksort')
        sorted_feats[j]=X[:,j][sorted_indxs[j]]
    # apply each rule
    rule_tx=np.zeros([X.shape[0],rule_lower_corners.shape[0]],dtype=np.int32)
    for r in np.arange(rule_lower_corners.shape[0]):
        viable_pts=set(np.arange(X.shape[0]))
        for j in np.arange(X.shape[1]):
            if rule_lower_corners[r,j]!=RULE_LOWER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], rule_lower_corners[r,j], side='right')
                viable_pts_this_feat=sorted_indxs[j][insert_pos:]
                if len(viable_pts_this_feat)>len(viable_pts):
                    viable_pts=viable_pts & set(viable_pts_this_feat) 
                else:
                    viable_pts=viable_pts & set(viable_pts_this_feat) 
            if rule_upper_corners[r,j]!=RULE_UPPER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], rule_upper_corners[r,j], side='right')
                viable_pts_this_feat=sorted_indxs[j][0:insert_pos]
                if len(viable_pts_this_feat)>len(viable_pts):
                    viable_pts=viable_pts & set(viable_pts_this_feat) 
                else:
                    viable_pts=viable_pts & set(viable_pts_this_feat)   
        if len(viable_pts)>0:
            rule_tx[np.asarray(list(viable_pts)),r] = 1
    return rule_tx
    # set banaive approachon, about 3X slower than cython naive approach
def apply_rules_sets_2(X,
        rule_lower_corners,
        rule_upper_corners,
        X_leaf_node_ids=None,
        node_rule_map=None):
    # sort X feats
    sorted_feats=dict()
    sorted_indxs=dict()
    for j in np.arange(X.shape[1]):
        sorted_indxs[j]= np.argsort(X[:,j], axis=-1, kind='quicksort')
        sorted_feats[j]=X[:,j][sorted_indxs[j]]
    # apply each rule
    rule_tx=np.zeros([X.shape[0],rule_lower_corners.shape[0]],dtype=np.int32)
    for r in np.arange(rule_lower_corners.shape[0]):
        viable_pts_dict=dict()
        viable_pts_dict[0]=np.arange(X.shape[0])
        viable_pts_keys=[0]
        viable_pts_lens=[len(viable_pts_dict[0])]
        #viable_pts=np.arange(X.shape[0])
        for j in np.arange(X.shape[1]):
            if rule_lower_corners[r,j]!=RULE_LOWER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], rule_lower_corners[r,j], side='right')
                viable_pts_dict[-j]=sorted_indxs[j][insert_pos:]
                viable_pts_keys=viable_pts_keys+[-j]
                viable_pts_lens=viable_pts_lens+[len(viable_pts_dict[-j])]
            if rule_upper_corners[r,j]!=RULE_UPPER_CONST:
                insert_pos=np.searchsorted(sorted_feats[j], rule_upper_corners[r,j], side='right')
                viable_pts_dict[j]=sorted_indxs[j][0:insert_pos]
                viable_pts_keys=viable_pts_keys+[j]
                viable_pts_lens=viable_pts_lens+[len(viable_pts_dict[j])]
        # join the sets, starting with smallest
        key_order=np.argsort(viable_pts_lens)
        key_order_=np.asarray(viable_pts_keys)[key_order]
        viable_pts_progressive=viable_pts_dict[key_order_[0]]
        for k in np.arange(1,len(key_order)):
            key=key_order_[k]
            #this_set=viable_pts_dict[k]
            viable_pts_progressive=np.intersect1d( viable_pts_progressive , viable_pts_dict[key])
        if len(viable_pts_progressive)>0:
            rule_tx[viable_pts_progressive,r] = 1        
    return rule_tx


def traverse_node_with_rule_c(node_id,
                       num_feats,
                       children_left,
                       children_right,
                       features,
                       thresholds, 
                       node_members,
                       node_members_count,
                       rule_id,
                       rule_upper_corners,
                       rule_lower_corners,
                       X,
                       out_rule_mask):
    # recurse on children 
    if children_left[node_id] != -1:
        feature = features[node_id]
        threshold = thresholds[node_id]
        left_node_id = children_left[node_id]
        right_node_id = children_right[node_id]
        # check if rule goes right
        if rule_upper_corners[feature]>threshold:
            rule_lower_corners1=rule_lower_corners.copy()
            if rule_lower_corners1[feature]<=threshold: # rule lower bound no longer needed
                rule_lower_corners1[feature]=RULE_LOWER_CONST
            traverse_node_with_rule_c(right_node_id, num_feats,
                   children_left,children_right,features,thresholds, node_members,node_members_count,rule_id,
                   rule_upper_corners.copy(),
                   rule_lower_corners1,
                   X,
                   out_rule_mask)  
        # check if rule goes left
        if rule_lower_corners[feature]<threshold:
            rule_upper_corners1=rule_upper_corners.copy()
            if rule_upper_corners1[feature]>=threshold: # rule lower bound no longer needed
                rule_upper_corners1[feature]=RULE_UPPER_CONST
            traverse_node_with_rule_c(left_node_id, num_feats,
                   children_left,children_right,features,thresholds, node_members,node_members_count,rule_id,
                   rule_upper_corners1,
                   rule_lower_corners.copy(),
                   X,
                   out_rule_mask)  
    else:  # a leaf node - check remaining rule
        num_pts=node_members_count[node_id]
        for j in np.arange(num_feats):
            lower_bound=rule_lower_corners[j]
            if lower_bound!=RULE_LOWER_CONST:
                for i in np.arange(num_pts):
                    if out_rule_mask[node_members[node_id,i],rule_id]!=2:
                        if X[node_members[node_id,i],j]<=lower_bound:
                            out_rule_mask[node_members[node_id,i],rule_id]=2
            upper_bound=rule_upper_corners[j]
            if upper_bound!=RULE_UPPER_CONST:
                for i in np.arange(num_pts):
                    if out_rule_mask[node_members[node_id,i],rule_id]!=2:
                        if X[node_members[node_id,i],j]>upper_bound:
                            out_rule_mask[node_members[node_id,i],rule_id]=2
        for i in np.arange(num_pts):
            if out_rule_mask[node_members[node_id,i],rule_id]==2:
                out_rule_mask[node_members[node_id,i],rule_id]=0
            else:
                out_rule_mask[node_members[node_id,i],rule_id]=1

def apply_rules_from_tree_c(X,
                            children_left,
                            children_right,
                            features,
                            thresholds,
                            node_members,
                            node_members_count, 
                            num_feats,
                            rule_upper_corners,
                            rule_lower_corners,
                            out_rule_mask):
    for rule_id in np.arange(rule_upper_corners.shape[0]):
        traverse_node_with_rule_c(0,
                       num_feats,
                       children_left,
                       children_right,
                       features,
                       thresholds, 
                       node_members,
                       node_members_count,
                       rule_id,
                       rule_upper_corners[rule_id,:].copy(),
                       rule_lower_corners[rule_id,:].copy(),
                       X,
                       out_rule_mask)
    
    
def apply_rules_tree(   X,
                        rule_lower_corners,
                        rule_upper_corners,
                        tree
                        ):
    X_leaf_node_ids = tree.apply(X, check_input=False).astype(np.int32)
    # cache X_leaf_node_ids for each node
    num_nodes=tree.tree_.node_count
    node_members=np.zeros([num_nodes,X.shape[0]],dtype=np.int32)
    node_member_count=np.zeros(num_nodes,dtype=np.int32)
    for inode in np.unique(X_leaf_node_ids):
        node_members_=np.where(X_leaf_node_ids==inode)[0]
        node_members[inode,0:len(node_members_)]=node_members_
        node_member_count[inode]=len(node_members_)
    rule_mask = np.zeros(
            [X.shape[0], rule_lower_corners.shape[0]], dtype=np.int32)
    apply_rules_from_tree_c(X,
                            tree.tree_.children_left,
                            tree.tree_.children_right,
                            tree.tree_.feature,
                            tree.tree_.threshold,
                            node_members,
                            node_member_count,
                            rule_upper_corners.shape[1],
                            rule_upper_corners,
                            rule_lower_corners,
                            rule_mask)
    return np.asarray(rule_mask, dtype=bool)

    
def apply_rules_tree_sorted(   X,
                        rule_lower_corners,
                        rule_upper_corners,
                        tree
                        ):

    X_leaf_node_ids = tree.apply(X, check_input=False).astype(np.int32)
    # cache X_leaf_node_ids for each node
    num_nodes=tree.tree_.node_count
    node_members=np.zeros([num_nodes,X.shape[0]],dtype=np.int32,order='C')
    node_member_count=np.zeros(num_nodes,dtype=np.int32,order='C')
    node_member_start=np.zeros(num_nodes,dtype=np.int32,order='C')
    X_by_node_sorted=np.zeros(X.shape,dtype=np.float64,order='C')
    X_by_node_sorted_idx=np.zeros(X.shape,dtype=np.int32,order='C')
    X_by_node_sorted_idx_posns=np.zeros(X.shape,dtype=np.int32,order='C')
    node_member_start_=0
    for inode in np.unique(X_leaf_node_ids):
        node_members_=np.where(X_leaf_node_ids==inode)[0]
        node_members_len=len(node_members_)
        node_members[inode,0:node_members_len]=node_members_
        node_member_count[inode]=node_members_len
        node_member_start[inode]=node_member_start_
        for j in np.arange(X.shape[1]):
            if node_members_len>=MIN_NODE_SIZE_FOR_SORTING_:
                X_by_node_sorted_idx[node_member_start_:node_member_start_+node_members_len,j]=node_members_[np.argsort(X[node_members_,j])]
            else:
                X_by_node_sorted_idx[node_member_start_:node_member_start_+node_members_len,j]=node_members_
            X_by_node_sorted[node_member_start_:node_member_start_+node_members_len,j]=X[X_by_node_sorted_idx[node_member_start_:node_member_start_+node_members_len,j],j] 
        node_member_start_=node_member_start_+node_members_len
    for j in np.arange(X.shape[1]):    
        X_by_node_sorted_idx_posns[X_by_node_sorted_idx[:,j],j]=np.arange(X.shape[0])
    
    rule_mask = np.zeros(
            [X.shape[0], rule_lower_corners.shape[0]], dtype=np.int32,order='C')
    apply_rules_from_tree_sorted_c(X.astype(np.float64),
                            X_by_node_sorted,
                            X_by_node_sorted_idx,
                            X_by_node_sorted_idx_posns,
                            tree.tree_.children_left.astype(np.int32),
                            tree.tree_.children_right.astype(np.int32),
                            tree.tree_.feature.astype(np.int32),
                            tree.tree_.threshold.astype(np.float64),
                            node_members,
                            node_member_count,
                            node_member_start,
                            np.int32(rule_upper_corners.shape[1]),
                            np.ascontiguousarray(rule_upper_corners,dtype=np.float64),
                            np.ascontiguousarray(rule_lower_corners,dtype=np.float64),
                            rule_mask)

    return np.asarray(rule_mask, dtype=bool)

    
def apply_rules(
        X,
        rule_lower_corners,
        rule_upper_corners,
        X_leaf_node_ids=None,
        node_rule_map=None,
        node_rule_feats_upper=None,
        node_rule_feats_lower=None):
    # NON SPARSE - no longer used, use sparse version for scaleable speed.
    #    rule_mask=np.zeros([X.shape[0],rule_lower_corners.shape[0]],dtype=int)
    #    apply_rules_c(np.asarray(X,dtype=np.float64),\
    #           np.asarray(rule_lower_corners,dtype=np.float64),
    #           np.asarray(rule_upper_corners,dtype=np.float64), rule_mask)
    #    rule_mask=np.asarray(rule_mask,dtype=bool)
    # SPARSE VERSION:
    if node_rule_feats_upper is not None:
        rule_mask = np.zeros(
            [X.shape[0], rule_lower_corners.shape[0]], dtype=np.int32)

        apply_rules_rule_feat_cache_c(
            X.astype(
                np.float64),
            rule_lower_corners,
            rule_upper_corners,
            X_leaf_node_ids,
            node_rule_map,
            node_rule_feats_upper,
            node_rule_feats_lower,
            rule_mask)
        rule_mask = np.asarray(rule_mask, dtype=bool)
    else:
        rule_mask=apply_rules_set_based(X,
            rule_lower_corners,
            rule_upper_corners)
    return np.asarray(rule_mask, dtype=bool)



def build_node_map_rule_feats(
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_lower_corners,
        rule_upper_corners):
    # NON SPARSE - no longer used, use sparse version for scaleable speed.
    #    map_=np.zeros([np.max(leaf_ids)+1,rule_upper_corners.shape[0]],
    #                  dtype=np.int32)-99
    #    for i_leaf in np.arange(len(leaf_ids)):
    #        leaf_id=leaf_ids[i_leaf] # rule 'i_leaf' is in this leaf
    #        leaf_rules_overlap=np.logical_and(np.all(leaf_lower_corners
    #           [i_leaf,:]<rule_upper_corners,axis=1),
    #           np.all(leaf_upper_corners[i_leaf,:]>
    #           rule_lower_corners,axis=1))
    #        rule_idxs=np.where(leaf_rules_overlap)[0]
    #        map_[leaf_id,0]=i_leaf # this was the base rule and
    #                               # is fully covered by this leaf
    #        rule_idxs=rule_idxs[np.nonzero(rule_idxs-i_leaf)]
    #        map_[leaf_id,1:len(rule_idxs)+1]=rule_idxs
    # SPARSE VERSION:
    rule_upper_corners_sparse = csr_matrix(
        rule_upper_corners - RULE_UPPER_CONST, dtype=np.float64)
    rule_lower_corners_sparse = csr_matrix(
        rule_lower_corners - RULE_LOWER_CONST, dtype=np.float64)
    n_leaves = np.max(leaf_ids) + 1  # leaf_lower_corners.shape[0] #
    map_c_ = np.zeros([n_leaves,
                       rule_upper_corners.shape[0]],
                      dtype=np.int32) - 99
    map_rule_feats_upper_c_ = np.zeros([n_leaves,
                                        rule_upper_corners.shape[0],
                                        leaf_lower_corners.shape[1]],
                                       dtype=np.int32) - 99
    map_rule_feats_lower_c_ = np.zeros([n_leaves,
                                        rule_upper_corners.shape[0],
                                        leaf_lower_corners.shape[1]],
                                       dtype=np.int32) - 99

    get_node_map_and_rule_feats_c(
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_lower_corners_sparse,
        rule_upper_corners_sparse,
        rule_upper_corners.shape[0],
        map_c_,
        map_rule_feats_upper_c_,
        map_rule_feats_lower_c_)
    return [map_c_, map_rule_feats_upper_c_, map_rule_feats_lower_c_]


def build_node_rule_map(
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_lower_corners,
        rule_upper_corners):
    # NON SPARSE - no longer used, use sparse version for scaleable speed.
    #    map_=np.zeros([np.max(leaf_ids)+1,rule_upper_corners.shape[0]],
    #                  dtype=np.int32)-99
    #    for i_leaf in np.arange(len(leaf_ids)):
    #        leaf_id=leaf_ids[i_leaf] # rule 'i_leaf' is in this leaf
    #        leaf_rules_overlap=np.logical_and(np.all(leaf_lower_corners
    #           [i_leaf,:]<rule_upper_corners,axis=1),
    #           np.all(leaf_upper_corners[i_leaf,:]>
    #           rule_lower_corners,axis=1))
    #        rule_idxs=np.where(leaf_rules_overlap)[0]
    #        map_[leaf_id,0]=i_leaf # this was the base rule and
    #                               # is fully covered by this leaf
    #        rule_idxs=rule_idxs[np.nonzero(rule_idxs-i_leaf)]
    #        map_[leaf_id,1:len(rule_idxs)+1]=rule_idxs
    # SPARSE VERSION:
    rule_upper_corners_sparse = csr_matrix(
        rule_upper_corners - RULE_UPPER_CONST, dtype=np.float64)
    rule_lower_corners_sparse = csr_matrix(
        rule_lower_corners - RULE_LOWER_CONST, dtype=np.float64)
    map_c_ = np.zeros([np.max(leaf_ids) + 1,
                       rule_upper_corners.shape[0]],
                      dtype=np.int32) - 99
    get_node_map_c(
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_lower_corners_sparse,
        rule_upper_corners_sparse,
        map_c_)
    return map_c_

 
def extract_rules_from_tree_cython(tree, num_feats, incr_feats, decr_feats):
    """Helper to turn a tree into as set of rules
    """
    num_nodes = tree.node_count
    leaf_ids = np.zeros([num_nodes], dtype=np.int32)-99
    leaf_values = np.zeros([num_nodes], dtype=np.float64)
    rule_upper_corners = np.ones(
        [num_nodes, num_feats], dtype=np.float64,order='C') * RULE_UPPER_CONST
    rule_lower_corners = np.ones(
        [num_nodes, num_feats], dtype=np.float64,order='C') * RULE_LOWER_CONST

    extract_rules_from_tree_c(tree.children_left.astype(np.int32),tree.children_right.astype(np.int32),tree.feature.astype(np.int32),tree.threshold.astype(np.float64), np.int32(num_feats), leaf_ids,rule_upper_corners,rule_lower_corners)
    for node_id in np.arange(num_nodes):
        leaf_values[node_id] = tree.value[node_id][0][0]
    # filter unused rules
    idx = leaf_ids != -99

    leaf_ids = leaf_ids[idx]
    leaf_values = leaf_values[idx]
    rule_upper_corners = rule_upper_corners[idx, :]
    rule_lower_corners = rule_lower_corners[idx, :]
    # store leaf definitions before montonising
    leaf_upper_corners = rule_upper_corners.copy()
    leaf_lower_corners = rule_lower_corners.copy()
    # monotonise if required
    if len(incr_feats) > 0:
        for i_mt in incr_feats - 1:
            rule_lower_corners[leaf_values < 0, i_mt] = RULE_LOWER_CONST
            rule_upper_corners[leaf_values > 0, i_mt] = RULE_UPPER_CONST
    if len(decr_feats) > 0:
        for i_mt in decr_feats - 1:
            rule_lower_corners[leaf_values > 0, i_mt] = RULE_LOWER_CONST
            rule_upper_corners[leaf_values < 0, i_mt] = RULE_UPPER_CONST

    # XXX eliminate this dist_feats usage, not needed or useful (too confusing)
    feats_idx =np.arange(rule_lower_corners.shape[1],dtype=np.int64)
    # XXX NEEDS TIDY UP: this re-allocation is needed to moves arrays from 
    # C-major to F-major. Fix the subsequent cython calls to always use C-major
    rule_upper_corners = rule_upper_corners[:, feats_idx]
    rule_lower_corners = rule_lower_corners[:, feats_idx]
    leaf_upper_corners = leaf_upper_corners[:, feats_idx]
    leaf_lower_corners = leaf_lower_corners[:, feats_idx]    
    return [
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_upper_corners,
        rule_lower_corners,
        feats_idx]
    

def extract_rules_from_tree(tree, num_feats, incr_feats, decr_feats):
    """Helper to turn a tree into as set of rules
    """
    num_nodes = tree.node_count
    leaf_ids = np.zeros([num_nodes], dtype=np.int32)
    leaf_values = np.zeros([num_nodes], dtype=np.float64)
    rule_upper_corners = np.ones(
        [num_nodes, num_feats], dtype=np.float64) * np.inf
    rule_lower_corners = np.ones(
        [num_nodes, num_feats], dtype=np.float64) * -np.inf

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       new_lower=None,
                       new_upper=None):
        if node_id == 0:
            new_lower = np.ones(num_feats) * RULE_LOWER_CONST
            new_upper = np.ones(num_feats) * RULE_UPPER_CONST
        else:
            if operator == +1:
                new_upper[feature] = threshold
            else:
                new_lower[feature] = threshold
        if tree.children_left[node_id] != TREE_LEAF:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, +1, threshold, feature,
                           new_lower.copy(), new_upper.copy())  # "<="

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, -1, threshold, feature,
                           new_lower.copy(), new_upper.copy())  # ">"
        else:  # a leaf node
            if node_id != 0:
                leaf_ids[node_id] = node_id
                leaf_values[node_id] = tree.value[node_id][0][0]
                rule_upper_corners[node_id, :] = new_upper
                rule_lower_corners[node_id, :] = new_lower
            else:  # the base node (0) is the only node!
                print('Warning: Tree only has one node! (i.e. the root node)')
            return None

    traverse_nodes()
    # filter unused rules
    idx = leaf_ids != 0

    leaf_ids = leaf_ids[idx]
    leaf_values = leaf_values[idx]
    rule_upper_corners = rule_upper_corners[idx, :]
    rule_lower_corners = rule_lower_corners[idx, :]
    # store leaf definitions before montonising
    leaf_upper_corners = rule_upper_corners.copy()
    leaf_lower_corners = rule_lower_corners.copy()
    # monotonise if required
    if len(incr_feats) > 0:
        for i_mt in incr_feats - 1:
            rule_lower_corners[leaf_values < 0, i_mt] = RULE_LOWER_CONST
            rule_upper_corners[leaf_values > 0, i_mt] = RULE_UPPER_CONST
    if len(decr_feats) > 0:
        for i_mt in decr_feats - 1:
            rule_lower_corners[leaf_values > 0, i_mt] = RULE_LOWER_CONST
            rule_upper_corners[leaf_values < 0, i_mt] = RULE_UPPER_CONST
    # filter unused features
    feats = np.any(np.vstack([rule_lower_corners != RULE_LOWER_CONST,
                              rule_upper_corners != RULE_UPPER_CONST]), axis=0)
    feats_idx = np.where(feats)[0]
    rule_upper_corners = rule_upper_corners[:, feats]
    rule_lower_corners = rule_lower_corners[:, feats]
    leaf_upper_corners = leaf_upper_corners[:, feats]
    leaf_lower_corners = leaf_lower_corners[:, feats]
    return [
        leaf_ids,
        leaf_values,
        leaf_lower_corners,
        leaf_upper_corners,
        rule_upper_corners,
        rule_lower_corners,
        feats_idx]


class RuleEnsemble(BaseEnsemble):
    """
    Internal class for storing n additive rule ensemble, where each rule r is
    defined by the region element-wise greater than rule_lower_corners[r,:] and
    less than rule_upper_corners[r,:], and has value rule_values[r]. Allows
    for a tree and node_rule_map to be included, which can speed up prediction
    by only checking rules that overlap with the leaf node a point belongs to.
    """

    def __init__(
            self,
            rule_lower_corners,
            rule_upper_corners,
            rule_values,
            dist_feats,
            tree=None,
            node_rule_map=None,
            intercept_=0.,
            node_rule_feats_upper=None,
            node_rule_feats_lower=None):
        self.rule_lower_corners = rule_lower_corners
        self.rule_upper_corners = rule_upper_corners
        self.rule_values = rule_values
        self.dist_feats = dist_feats
        self.tree = tree
        self.node_rule_map = node_rule_map
        self.intercept_ = intercept_
        self.node_rule_feats_upper = node_rule_feats_upper
        self.node_rule_feats_lower = node_rule_feats_lower

    def _validate_X_predict(self, X, check_input=True):
        if len(X.shape) == 1:
            X = X.reshape([-1, X.shape[0]])
        return X

    def decision_function(self, X):
        X = self._validate_X_predict(X, check_input=True)
        #print(self.intercept_)
        res = np.zeros(X.shape[0]) + self.intercept_
        if self.tree is not None:
            X_leaf_node_ids = None # no longer used 
            if self.node_rule_feats_upper is not None:
                rule_mask = apply_rules(
                        X[:, self.dist_feats],
                        self.rule_lower_corners,
                        self.rule_upper_corners,
                        X_leaf_node_ids=X_leaf_node_ids,
                        node_rule_map=self.node_rule_map,
                        node_rule_feats_upper=self.node_rule_feats_upper,
                        node_rule_feats_lower=self.node_rule_feats_lower)

            else:
                rule_mask = apply_rules(X[:,
                                          self.dist_feats],
                                        self.rule_lower_corners,
                                        self.rule_upper_corners,
                                        X_leaf_node_ids=X_leaf_node_ids,
                                        node_rule_map=self.node_rule_map)
        else:

            rule_mask = apply_rules(X[:, self.dist_feats],
                                    self.rule_lower_corners,
                                    self.rule_upper_corners)
        for i_rule in np.arange(len(self.rule_values)):
            res += (rule_mask[:, i_rule].astype(float) *
                    self.rule_values[i_rule])
        return res

    def predict_proba(self, X):
        res = self.decision_function(X)
        return res

    def predict(self, X):
        return np.sign(self.predict_proba(X))


class BaseMonoGradientBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Abstract base class for Gradient Boosting. """

    @abstractmethod
    def __init__(
            self,
            loss,
            learning_rate,
            n_estimators,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_depth,
            min_impurity_decrease,
            min_impurity_split,
            init,
            subsample,
            max_features,
            random_state,
            alpha=0.9,
            verbose=0,
            max_leaf_nodes=None,
            warm_start=False,
            presort='auto',
            incr_feats=[],
            decr_feats=[],
            coef_calc_type='bayes',
            rule_feat_caching=False):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.incr_feats = incr_feats
        self.decr_feats = decr_feats
        self.coef_calc_type = coef_calc_type
        self.rule_feat_caching = rule_feat_caching

    @property
    def mt_feats(self):
        if len(self.incr_feats) == 0 and len(self.decr_feats) == 0:
            return np.asarray([]).astype(int)
        else:
            return np.asarray(list(self.incr_feats) +
                              list(self.decr_feats)).astype(int) - 1

    @property
    def incr_feats(self):
        return self.incr_feats_

    @incr_feats.setter
    def incr_feats(self, x):
        self.incr_feats_ = np.asarray(x)

    @property
    def decr_feats(self):
        return self.decr_feats_

    @decr_feats.setter
    def decr_feats(self, x):
        self.decr_feats_ = np.asarray(x)

    def _validate_X_predict(self, X, check_input=True):
        """ Dummy function required to be used as RF component estimator"""
        return X

    def _fit_stage(self, i, X, y, y_pred, sample_weight, sample_mask,
                   random_state, X_idx_sorted, X_csc=None, X_csr=None):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == np.bool
        loss = self.loss_
        original_y = y

        k_max = 1 if loss.K <= 2 else loss.K - 1
        for k in range(k_max):
            if loss.is_multi_class:
                y = np.array(original_y > k, dtype=np.float64)

            residual = loss.negative_gradient(y, y_pred, k=k,
                                              sample_weight=sample_weight)

            tree = DecisionTreeRegressorSklearn(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                presort=self.presort)  

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            if X_csc is not None:
                tree.fit(X_csc, residual, sample_weight=sample_weight,
                         check_input=False, X_idx_sorted=X_idx_sorted)
            else:
                tree.fit(X, residual, sample_weight=sample_weight,
                         check_input=False, X_idx_sorted=X_idx_sorted)
                
            X_use = X_csr if X_csr is not None else X

            X_leaf_node_ids = None # no longer used under set regime tree.apply(X_use, check_input=False).astype(np.int32)

            # extract monotone rules
            [leaf_ids,
             leaf_values,
             leaf_lower_corners,
             leaf_upper_corners,
             rule_upper_corners,
             rule_lower_corners,
             dist_feats] = extract_rules_from_tree_cython(tree.tree_,
                                                   X.shape[1],
                                                   self.incr_feats,
                                                   self.decr_feats)
     
            X_leaf_node_ids = None # no longer used under set regime 
            
            # build node rule map
            if len(leaf_ids) > 0:
                if self.rule_feat_caching:
                    [node_rule_map, node_rule_feats_upper,
                     node_rule_feats_lower] = build_node_map_rule_feats(
                        leaf_ids,
                        leaf_values,
                        leaf_lower_corners,
                        leaf_upper_corners,
                        rule_lower_corners,
                        rule_upper_corners)
                else:
                    node_rule_map=None
                    node_rule_feats_upper = None
                    node_rule_feats_lower = None
            else:
                node_rule_map = None
                node_rule_feats_upper = None
                node_rule_feats_lower = None

            # update tree leaves
            # only use intercept for single trees and logistic solve
            intercept = (self.coef_calc_type ==
                         'logistic' and self.n_estimators == 1)
            adjust_intercept=(self.coef_calc_type ==
                         'bayes' and self.n_estimators == 1)
            #if self.mt_type=='global':
                
            if node_rule_feats_upper is None:
                # certified version of apply_rules
                rule_mask = apply_rules(
                    X_use[:, dist_feats],
                    rule_lower_corners,
                    rule_upper_corners,
                    X_leaf_node_ids,
                    node_rule_map)
            else:
                X_leaf_node_ids = tree.apply(X_use, check_input=False).astype(np.int32)
        
                rule_mask = apply_rules(
                    X_use[:, dist_feats],
                    rule_lower_corners,
                    rule_upper_corners,
                    X_leaf_node_ids,
                    node_rule_map,
                    node_rule_feats_upper,
                    node_rule_feats_lower)
            intercept_ = loss.update_terminal_rules(
                rule_mask,
                leaf_values,
                y,
                residual,
                y_pred,
                sample_weight,
                learning_rate=self.learning_rate,
                k=k,
                logistic_intercept=intercept,
                adjust_intercept_=adjust_intercept)
            if self.coef_calc_type=='bayes' and self.n_estimators==1:
                self.init_ = ZeroEstimator(n_classes=2 if self.loss_.K <= 2 else self.loss_.K) # use recalculated adjusted intercept
                self.init_.fit(X, y.astype(np.int32), sample_weight)
            self.estimators_[i, k] = RuleEnsemble(rule_lower_corners,
                                                  rule_upper_corners,
                                                  leaf_values,
                                                  dist_feats,
                                                  tree,
                                                  node_rule_map,
                                                  intercept_,
                                                  node_rule_feats_upper,
                                                  node_rule_feats_lower)
            
        return y_pred

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS or
                self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (BinomialDeviance)
            self.loss_ = loss_class(
                self.n_classes_,
                coef_calc_type=self.coef_calc_type,
                incr_feats=self.incr_feats,
                decr_feats=self.decr_feats)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

            if self.loss in ('huber', 'quantile'):
                self.loss_ = loss_class(self.n_classes_, self.alpha)
            else:
                self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            if isinstance(self.init, six.string_types):
                if self.init not in INIT_ESTIMATORS:
                    raise ValueError('init="%s" is not supported' % self.init)
            else:
                if (not hasattr(self.init, 'fit') or
                        not hasattr(self.init, 'predict')):
                    raise ValueError("init=%r must be valid BaseEstimator "
                                     "and support both fit and "
                                     "predict" % self.init)

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        if self.coef_calc_type == 'logistic' and self.n_estimators == 1:
            self.init_ = ZeroEstimator()
        elif self.init is None:
            self.init_ = self.loss_.init_estimator()
        elif isinstance(self.init, six.string_types):
            self.init_ = INIT_ESTIMATORS[self.init]()
        else:
            self.init_ = self.init
        num_K = 1 if self.loss_.K <= 2 else self.loss_.K - 1
        self.estimators_ = np.empty((self.n_estimators, num_K),
                                    dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        num_k = 1 if self.loss_.K <= 2 else self.loss_.K - 1
        self.estimators_.resize((total_n_estimators, num_k))
        self.train_score_.resize(total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_.resize(total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self, 'estimators_')

    @property
    @deprecated("Attribute n_features was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def n_features(self):
        return self.n_features_

    def fit(self, X, y, sample_weight=None, monitor=None, check_input=False):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
            Returns self.
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        if len(y.shape) > 1:
            # when called from RandomForestClassifier, y is converted to a
            # (n,1) column vector that generates a warning in the next line,
            # unless we ravel it here
            y = y.ravel()
        X, y = check_X_y(
            X, y, accept_sparse=[
                'csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
            check_consistent_length(X, y, sample_weight)
            y = self._validate_y(y)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            check_consistent_length(X, y, sample_weight)
            y = self._validate_y(y)
            # Attempted removing sample_weight==0 training points
            # BUT causes errors for multi-class with small minority class(es)
            # (e.g. SWD dataset)
            # sample_weight_mask=sample_weight>0
            # X=X[sample_weight_mask,:]
            # y=y[sample_weight_mask]
            # sample_weight=sample_weight[sample_weight_mask]

        random_state = check_random_state(self.random_state)
        self._check_params()
        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model - FIXME make sample_weight optional
            self.init_.fit(X, y, sample_weight)

            # init predictions
            y_pred = self.init_.predict(X)
            begin_at_stage = 0
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            y_pred = self._decision_function(X)
            self._resize_state()

        X_idx_sorted = None
        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if presort == 'auto' and issparse(X):
            presort = False
        elif presort == 'auto':
            presort = True

        if presort:
            if issparse(X):
                raise ValueError(
                    "Presorting is not supported for sparse matrices.")
            else:
                X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                                 dtype=np.int32)

        # create mt_feat_types
        self.mt_feat_types = np.zeros(X.shape[1],dtype=np.int32)
        if len(self.incr_feats)>0:
            self.mt_feat_types [self.incr_feats-1]=1
        if len(self.decr_feats)>0:
            self.mt_feat_types [self.decr_feats-1]=-1
        
        # fit the boosting stages
        n_stages = self._fit_stages(X, y, y_pred, sample_weight, random_state,
                                    begin_at_stage, monitor, X_idx_sorted)
        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        return self

    def _fit_stages(self, X, y, y_pred, sample_weight, random_state,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      y_pred[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_weight,
                                     sample_mask, random_state, X_idx_sorted,
                                     X_csc, X_csr)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             y_pred[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y[~sample_mask],
                                          y_pred[~sample_mask],
                                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, y_pred, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break
        return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _init_decision_function(self, X):
        """Check input and compute prediction of ``init``. """
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        score = self.init_.predict(X).astype(np.float64)
        return score

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.

        K = self.estimators_.shape[1]
        score = self._init_decision_function(X)  # np.zeros([X.shape[0],K])
        for i in range(self.estimators_.shape[0]):
            for k in range(K):
                score[:, k] += self.learning_rate * \
                    self.estimators_[i, k].decision_function(X)
        return score


    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        self._check_initialized()

        total_sum = np.zeros((self.n_features_, ), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def _validate_y(self, y):
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators, n_classes]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves


class MonoGradientBoostingClassifier(
        BaseMonoGradientBoosting,
        ClassifierMixin):
    """Monotone Gradient Boosting for classification.

    This is a high performance implementation of monotone classifier proposed
    in Bartley et al. 2017. It behaves identically to sci-kit learn's
    GradientBoostingClassifier with the following exceptions:
        1. The interface has three additional parameters for the constructor:
        ``incr_feats``, ``decr_feats``, and ``coef_calc_type``. The first two
        simply define which features are required to be monotone increasing
        (decreasing), and the last defines which solution to use for
        coefficient recalculation. See below for more details.
        2. After fitting each stage, the tree leaf rules are made
        'monotone compliant', and the coefficients re-calculated are
        constrained to ensure monotonicity.
        3. Logistic regression and Naive Bayesian techniques are offered as
        solutions to coefficient re-calculation method, in addition to the
        single Newton step used in GB.
        4. Multi-class capability is implemented using the monotone compliant
        ensembling method described in Kotlowski and Slowinski 2013.

    This implementation makes heavy re-use of the original sci-kit learn code,
    but substantial additions have been made to efficiently implement the
    algorithm.

    Read more in the README.

    Parameters
    ----------
    incr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone increasing impact on the resulting class.

    decr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone decreasing impact on the resulting class.

    coef_calc_type : string
        Determines how the rule coefficients are calculated. Allowable values:
        'logistic': L2 regularised logistic regression. Slower.
        'boost' DEFAULT: A single Newton step approximation is used. Fast, and
        generally best.
        'bayesian': Assumes conditional indpendence between rules and
        calculates coefficients as per Naive bayesian classification. Fast
        with good results.
    rule_feat_caching : bool
        If True, caches the features that distinguish the each rule from each
        leaf. This can make prediction faster, but at the cost of increased
        memory usage and sometimes slightly slower fit().
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm. ONLY DEVIANCE IS IMPLEMENTED.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init : BaseEstimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators,
                                                             ``loss_.K``]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    C. Bartley, W. Liu, and M. Reynolds.
    Fast & Perfect Monotone Random Forest Classication, 2017

    W. Kotlowskiand R. Slowinski.
    On Nonparametric Ordinal Classification with Monotonicity Constraints, 2013

    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    """

    _SUPPORTED_LOSS = ('deviance')  # , 'exponential')

    def __init__(self, incr_feats=[], decr_feats=[], coef_calc_type='bayes',
                 rule_feat_caching=False,
                 loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto'):

        super(MonoGradientBoostingClassifier, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start,
            presort=presort,
            incr_feats=incr_feats,
            decr_feats=decr_feats,
            coef_calc_type=coef_calc_type)

    def _validate_y(self, y):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        X_preproc = X.copy()
        score = self._decision_function(X_preproc)
        if score.shape[1] == 1:
            return score.ravel()
        return score

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        for dec in self._staged_decision_function(X):
            # no yield from in Python2.X
            yield dec

    def predict(self, X, check_input=False):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        score = self.decision_function(X)
        decisions = self.loss_._score_to_decision(score)
        return self.classes_.take(decisions, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for score in self._staged_decision_function(X):
            decisions = self.loss_._score_to_decision(score)
            yield self.classes_.take(decisions, axis=0)

    def predict_proba(self, X, check_input=False):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Reverse deccreasing features here for simplicity

        score = self.decision_function(X)
        try:
            return self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)

    def predict_log_proba(self, X, check_input=False):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        try:
            for score in self._staged_decision_function(X):
                yield self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)


##################################################
#   START CONSTRAINED LOGISTIC REGRESSION CODE   #
##################################################
class ConstrainedLogisticRegression:
    """A bounds constrained logistic regression with solution offset.

    Required to enable ``coef_calc_type``='logistic' option for
    ``MonoGradientBoostingClassifier`` - because sci-kit learn
    ``LogisticRegression`` does not allow bounds constraints or solution
    offset. It is included here to reduce dependencies.

    This is a simple implementation of logistic regression with L2
    regularisation, bounds constraints on the coefficients, and the ability
    to accept an offset on the solution. It is solved using the scipy
    conjugate gradient Newton method ``fmin_tnc``.

    Parameters
    ----------
    C : float, 0 < C <= 1.0
        Empirical error weight, inverse of lambda (regularisation strength).

    fit_intercept : boolean
        True to fit an intercept.

    random_state : random_state
        Random seed.

    solver : 'newton-cg'
        Uses scipy.opt.fmin_tnc (only implemented option)

    incr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone increasing impact on the resulting class.

    decr_feats : array-like
        The one-based array indices of the columns in X that should only have
        a monotone decreasing impact on the resulting class.

    regularise_intercept : boolean
        True to regularise the intercept.

    standardize : boolean
        True to standardise the predictors before regression. Prevents bias
        due to regularisation scale.

    penalty : string, optional (default='l2')
        The regularisation penalty. 'l1' is not guaranteed to work very well
        because the newton solver relies on smoothness.


    """

    def __init__(self, C=1.0, fit_intercept=True, random_state=None,
                 solver='newton-cg', incr_feats=[], decr_feats=[],
                 regularise_intercept=False, standardize=False, penalty='l2'):
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.solver = solver
        self.C = C
        self.incr_feats = np.asarray(incr_feats)
        self.decr_feats = np.asarray(decr_feats)
        self.has_mt_feats = len(incr_feats) > 0 or len(decr_feats) > 0
        self.regularise_intercept = regularise_intercept \
            if self.fit_intercept else False
        self.standardize = standardize
        self.penalty = penalty

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        offset : array-like, shape = [n_samples] (optional)
            The solution offset for each training point.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = np.sort(np.unique(y))
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.ravel(sample_weight)
        X_ = X.copy()
        y = np.ravel(y)
        y[y == 0] = -1
        if self.standardize:
            stds = np.sqrt(
                np.average(
                    (X_ -
                     np.average(
                         X_,
                         axis=0,
                         weights=sample_weight))**2,
                    axis=0,
                    weights=sample_weight))  # Fast and numerically precise
            X_[:, stds != 0] = X_[:, stds != 0] / stds[stds != 0]
        regularise_intercept_ = self.regularise_intercept \
            if self.fit_intercept else True

        # Solve
        if self.solver == 'newton-cg':
            coef_limits = [
                (0 if j in self.incr_feats -
                 1 else None,
                 0 if j in self.decr_feats -
                 1 else None) for j in np.arange(
                    X_.shape[1])]
            coef = nnlr(
                X_,
                y,
                sample_weight,
                self.C,
                coef_limits,
                regularise_intercept_,
                1. if self.penalty == 'l2' else 0.,
                self.fit_intercept,
                offset)
            if self.fit_intercept:
                self.intercept_ = np.asarray([coef[-1]])
                self.coef_ = np.asarray([coef[0:len(coef) - 1]])
            else:
                self.intercept_ = np.asarray([0.])
                self.coef_ = np.asarray([coef])
        elif self.solver == 'two-pass':
            pass
        if self.standardize:
            self.coef_[0][stds != 0] = self.coef_[
                0][stds != 0] / stds[stds != 0]

    def predict(self, X):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        proba = self.predict_proba(X)

        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")
        prob1 = 1. / \
            (1 + np.exp(np.dot(self.coef_, -X.T).reshape(-1, 1) -
                        self.intercept_))
        probs = np.hstack([1 - prob1, prob1])
        return probs

    def decision_function(self, X):
        """Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_log_proba(self, X):
        """Log of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


#def calc_deriv(y,f):
#    np.seterr(all='raise')
#    try:
#        res=-2*y/(np.exp(2*y*f)+1)
#    except FloatingPointError as err:
#        print('here')
#    if np.any(np.isnan(res)):
#        print('sdfffsdf')
#    return res
#
#def calc_deriv_2(y,f):
#    try:
#        return 4*y**2*np.exp(2*y*f)/(1+np.exp(2*y*f))**2
#    except FloatingPointError as err:
#        print('here')
#
#def calc_newton_step(y,f,sample_weight):
#    return np.sum(sample_weight*calc_deriv(y,f))/np.sum(sample_weight*calc_deriv_2(y,f))

def adjust_intercept(intercept_,
                          y_pred,
                          y,
                          sample_weight,
                          coefs):
    y_=np.asarray(np.where(y==1,1,-1),dtype=np.float64)
    y_min=np.min(np.where(sample_weight>0,y_,100))
    if y_min==np.max(np.where(sample_weight>0,y_,-100)): # all the same class!
        return -1. if y_min<0 else +1.
    else:
        intercept_old=99
        intercept_new=intercept_
        f_base=y_pred-intercept_
        mx_=50.
        f_base=np.where(f_base>mx_,mx_,f_base)
        f_base=np.where(f_base<-mx_,-mx_,f_base)
        
        iters_=0
        while np.abs(intercept_old-intercept_new)>1e-3 and iters_<15:
            #incr=calc_newton_step(y_,f_base+intercept_new,sample_weight)
            incr=calc_newton_step_c(y_,f_base+intercept_new,sample_weight)
            if incr>1e0: # for stability under near floating point overflow errors
                incr=1e0
            elif incr<-1e0:
                incr=-1e0
            intercept_old=intercept_new
            intercept_new=intercept_new-incr
            iters_=iters_+1
        return intercept_new

def _logistic_grad(
        w,
        X,
        y,
        alpha,
        sample_weight=None,
        regularise_intercept=True,
        offset=None):
    """Computes the logistic loss and gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    if offset is None:
        offset = np.zeros(len(y))
    n_samples, n_features = X.shape
    grad = np.empty_like(w)
    intercept = w[-1] if (regularise_intercept and len(w) > X.shape[1]) else 0.
    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    z = expit(yz + y * offset)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum() + alpha * intercept
    return grad


def _logistic_loss(
        w,
        X,
        y,
        alpha,
        sample_weight=None,
        regularise_intercept=True,
        offset=None):
    """Computes the logistic loss.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Logistic loss.
    """
    if offset is None:
        offset = np.zeros(len(y))
    intercept = w[-1] if (regularise_intercept and len(w) > X.shape[1]) else 0.
    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    n_samples=len(y)
    
    out = -np.sum(sample_weight * _log_logistic_sigmoid(n_samples,  yz + y * offset)) + .5 * \
        alpha * (np.dot(w, w) + intercept**2)
    
   
    

    return out


def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    Returns
    -------
    w : ndarray, shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Unchanged.
    yz : float
        y * np.dot(X, w).
    """
    c = 0.
    if w.size == X.shape[1] + 1: # has intercept
        c = w[-1]
        w = w[:-1]
    z = safe_sparse_dot(X, w) + c
    yz = y * z
    
    return w, c, yz


def nnlr(X, y, sample_weight, C, coef_limits, regularise_intercept, alpha,
         fit_intercept, offset):
    """
    Non-negative Logistic Regression with L2 regularizer
    """

    N = X.shape[1]

    def J(theta):
        return _logistic_loss(theta, X, y, 1. / C,
                              sample_weight=sample_weight,
                              regularise_intercept=regularise_intercept,
                              offset=offset)

    def J_grad(theta):
        return _logistic_grad(theta, X, y, 1. / C,
                              sample_weight=sample_weight,
                              regularise_intercept=regularise_intercept,
                              offset=offset)
    if fit_intercept:
        theta0 = 0.0 * sp.ones(N + 1)
        coef_limits = coef_limits + [(None, None)]
    else:
        theta0 = 0.0 * sp.ones(N)
    x, nfeval, rc = opt.fmin_tnc(J, theta0, fprime=J_grad, bounds=coef_limits,
                                 disp=0, messages=0)
    return x

################################################
#   END CONSTRAINED LOGISTIC REGRESSION CODE   #
################################################


