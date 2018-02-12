"""Forest of monotone trees-based ensemble methods

This module contains methods for fitting monotone tree ensembles for
classification. Monotonicity in the requested features is achieved using the
technique from Bartley C., Liu W., and Reynolds M. 2017, ``Fast & Perfect
Monotone Random Forest Classification``, available
here_(http://staffhome.ecm.uwa.edu.au/~19514733/). Multi-class classification
is implemented using the monotone ensembling procedure from Kotlowski W, and
Slowinski R., 2013 ``On nonparametric ordinal classification with monotonicity
constraints``, IEEE Transactions on Knowledge and Data Engineering, vol 25,
no. 11, 2576--2589.

Currently only RandomForests is implemented, and this implementation is
dependent on (and inherits from) sci-kit learn's ``ForestClassifier``.


"""

# Author: Christopher Bartley <chris@bartleys.net>
#          Inherits from sci-kit learn's Random Forest, written by:
#          Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble.forest import ForestClassifier, check_array
from sklearn.ensemble.forest import _generate_unsampled_indices, DTYPE, warn
from monoensemble import MonoGradientBoostingClassifier
import numpy as np


class MonoRandomForestClassifier(ForestClassifier):
    """A fast and perfect monotone random forest classifier.
    This is a high performance implementation of monotone random forest
    classifier proposed in Bartley et al. 2017. It behaves identically to
    sci-kit learn's GradientBoostingClassifier with the following exceptions:
        1. The interface has three additional parameters for the constructor:
        ``incr_feats``, ``decr_feats``, and ``coef_calc_type``. The first two
        simply define which features are required to be monotone increasing
        (decreasing), and the last defines which solution to use for
        coefficient recalculation. See below for more details.
        2. After fitting each tree, the tree leaf rules are made
        'monotone compliant', and the coefficients re-calculated are
        constrained to ensure monotonicity (see paper for details).
        3. Three coefficient re-calculation methods are offered: Logistic
        regression, Naive Bayesian techniques, and gradient boosting's
        single Newton step.
        4. Multi-class capability is implemented using the monotone compliant
        ensembling method described in Kotlowski and Slowinski 2013.

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

    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

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
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. versionadded:: 0.18

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    .. [2] http://arogozhnikov.github.io/2016/07/12/secret-of-ml.html:
           Note that using mono_gradient_boosting works even though it uses MSE
           rather than gini, because for binary 0/1 classification the results
           are the same!
    .. [3] C. Bartley, W. Liu, and M. Reynolds., Fast & Perfect Monotone
           Random Forest Classication, 2017
    .. [4] W. Kotlowskiand R. Slowinski., On Nonparametric Ordinal
           Classification with Monotonicity Constraints, 2013


    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """

    def __init__(self,
                 incr_feats=[],
                 decr_feats=[],
                 coef_calc_type='bayes',
                 rule_feat_caching=False,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None
                 ):
        super(
            MonoRandomForestClassifier,
            self).__init__(
            base_estimator=MonoGradientBoostingClassifier(
                n_estimators=1,
                learning_rate=1,
                criterion='friedman_mse'),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
                "incr_feats",
                "decr_feats",
                "coef_calc_type",
                "rule_feat_caching"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.incr_feats = incr_feats
        self.decr_feats = decr_feats
        self.coef_calc_type = coef_calc_type
        self.rule_feat_caching = rule_feat_caching

    def predict(self, X):
        """Predict class for X.

        NOTE: We need to override parent method's predict() so we can
        allow for Kotlowski and Slowinski style monotone ensembling.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            if self.n_classes_ <= 2:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)
            else:
                class_index = np.sum((proba > 0.5).astype(np.int), axis=1)
                return self.classes_.take(class_index, axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                if self.n_classes_ <= 2:
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[k], axis=1), axis=0)
                else:
                    class_index = np.sum(
                        (proba[k] > 0.5).astype(
                            np.int), axis=1)
                    predictions[:, k] = self.classes_[
                        k].take(class_index, axis=0)

            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.

        NOTE: We need to override parent method's predict() so we can
        allow for Kotlowski and Slowinski style monotone ensembling.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the
        same class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        n_classes = self.n_classes_
        if self.n_classes_ > 2:
            # need to do this to allow for Kotloski and Slowinski style
            # monotone ensembling which has has k-1 estimators rather than k
            self.n_classes_ = n_classes - 1
        proba = super().predict_proba(X)
        self.n_classes_ = n_classes  # return to normal
        return proba

    def get_leaf_counts(self, only_count_non_zero=True):
        numtrees = np.int(self.get_params()['n_estimators'])
        num_leaves = np.zeros(numtrees, dtype='float')
        for itree in np.arange(numtrees):
            tree = self.estimators_[itree].estimators_[0, 0].tree.tree_
            n_nodes = tree.node_count
            children_left = tree.children_left
            children_right = tree.children_right
            node_depth = np.zeros(shape=n_nodes)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1

                # If we have a test node
                if node_is_leaf(
                        tree,
                        node_id,
                        only_count_non_zero=only_count_non_zero):
                    is_leaves[node_id] = True
                elif node_is_leaf(tree, node_id, only_count_non_zero=False):
                    is_leaves[node_id] = False
                else:
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))

            num_leaves[itree] = np.sum(is_leaves)
        return num_leaves

    # override this because the multi-class form has K-1 values not K due
    # to monotone ensembling
    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')
        if self.n_classes_[0] > 2:
            n_classes_ = list(
                np.asarray(
                    self.n_classes_) -
                1)  # CHANGED TO K-1
        else:
            n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            if self.n_classes_[0] <= 2:
                oob_score += np.mean(y[:, k] ==
                                     np.argmax(predictions[k], axis=1), axis=0)
            else:
                class_index = np.sum(
                    (predictions[k] > 0.5).astype(
                        np.int), axis=1)
                oob_score += np.mean(y[:, k] == class_index, axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_


def node_is_leaf(tree, node_id, only_count_non_zero=False):
    if only_count_non_zero:
        return tree.children_left[node_id] == tree.children_right[node_id] and\
            not np.all(np.asarray(tree.value[node_id][0]) == 0.)
    else:
        return tree.children_left[node_id] == tree.children_right[node_id]
