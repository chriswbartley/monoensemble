## monoensemble
<!---
[![Build Status](https://travis-ci.org/chriswbartley/monoensemble.svg?branch=master)](https://travis-ci.org/chriswbartley/monoensemble)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/chriswbartley/monoensemble)](https://ci.appveyor.com/project/chriswbartley/monoensemble)
[![RTD Status](https://readthedocs.org/projects/monoensemble/badge/?version=latest
)](https://readthedocs.org/projects/monoensemble/badge/?version=latest)
-->

This package implements monotone versions of the Random Forest and Gradient Boosting classification algorithms. The two provided algorithms (MonoRandomForestClassifier and MonoGradientBoostingClassifier) are fast, achieve perfect monotonicity, and have multi-class capability. In addition, as well as *partial* monotonicity capability (i.e. the ability to specify both monotone and non-monotone features), they are based on the corresponding `scit-kit learn` classifiers and provide all the associated functionality (such as e.g. out-of-box performance estimates). The theory is described in Bartley C., Liu W., Reynolds M., 2017, *Fast & Perfect Monotone Random Forest
Classication.* prepub, available [here](http://staffhome.ecm.uwa.edu.au/~19514733/). 

### Code Example
First we define the monotone features, using the corresponding one-based `X` array column indices:
```
incr_feats=[6,9]
decr_feats=[1,8,13]
```
The specify the hyperparameters (see original paper for explanation):
```
# Ensure you have a reasonable number of trees
n_estimators=200
mtry = 3
```
And initialise and solve the classifier using `scikit-learn` norms:
```
clf = mono_forest.MonoRandomForestClassifier(n_estimators=n_estimators,
                                             max_features=mtry,
                                             incr_feats=incr_feats,
                                             decr_feats=decr_feats)
clf.fit(X, y)
y_pred = clf.predict(X)
```	
Of course usually the above will be embedded in some estimate of generalisation error such as out-of-box (oob) score or cross-validation.

### Documentation

For more examples see [the documentation](http://monoensemble.readthedocs.io/en/latest/index.html).

### Installation

Linux:

```
pip install monoensemble
```

Windows:

```
conda install -c chriswbartley monoensemble
```

### Documentation

Documentation is provided [here](http://monoensemble.readthedocs.io/en/latest/index.html).

### Contributors

Pull requests welcome! Notes:
 - We use the
[PEP8 code formatting standard](https://www.python.org/dev/peps/pep-0008/), and
we enforce this by running a code-linter called
[`flake8`](http://flake8.pycqa.org/en/latest/) during continuous integration.
 - Continuous integration is used to run the tests in `/monoensemble/tests/test_monoensemble.py`, using [Travis](https://travis-ci.org/chriswbartley/monoensemble.svg?branch=master) (Linux) and [Appveyor](https://ci.appveyor.com/api/projects/status/github/chriswbartley/monoensemble) (Windows).
 
### License
BSD 3 Clause, Copyright (c) 2017, Christopher Bartley
All rights reserved.
