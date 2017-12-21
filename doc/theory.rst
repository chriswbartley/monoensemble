
Theory
========================

The monotone classification algorithms implemented here are described in the paper paper [bartley2017]_. The component trees are converted to monotone compliant rule sets in two stages. First the leaf 'rules' are made monotone compliant (i.e. made to comply with Theorem 3.1 and Lemma 4.1 in the paper). Then the rule coefficients are recalculated using one of three techniques: logistic regression, Naive Bayesian classification, or boosting (single Newton step). The result is a classifier that is perfectly monotone in the requested features, and very fast. 

Please refer to the paper for more detail.

.. DELETE_THIS_TO_USEmath::
    F(\textbf{x})=sign(a_0 + \sum_{m=1}^{M}a_m f_m(\textbf{x}))




.. [bartley2017] Bartley C., Liu W., Reynolds M. (2017). Fast & Perfect Monotone Random Forest Classication. prepub, http://staffhome.ecm.uwa.edu.au/~19514733/

