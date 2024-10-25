PyGRidge Scikit-learn Integration Guide
=====================================

This document describes how PyGRidge integrates with scikit-learn's API and conventions.

Base Classes and Mixins
--------------------

PyGRidge estimators inherit from scikit-learn's base classes:

- ``BaseEstimator``: Provides get_params/set_params functionality
- ``TransformerMixin``: For classes implementing transform/fit_transform
- ``RegressorMixin``: For regression estimators

API Conventions
------------

Our estimators follow scikit-learn's API conventions:

- ``fit(X, y)`` : Fits the model to training data
- ``predict(X)`` : Makes predictions on new data
- ``transform(X)`` : Transforms input data (for transformer classes)
- ``fit_transform(X, y)`` : Convenience method combining fit and transform

Parameter Naming
-------------

We follow scikit-learn's parameter naming conventions:

- ``X`` : Feature matrix (n_samples, n_features)
- ``y`` : Target values
- ``sample_weight`` : Optional sample weights
- ``groups`` : For group-based estimators

Input Validation
-------------

PyGRidge uses scikit-learn's validation tools:

- ``check_X_y()`` : Validate X and y arrays
- ``check_array()`` : Validate single array
- ``check_is_fitted()`` : Verify estimator is fitted

Examples
-------

See the ``examples/`` directory for practical demonstrations of:

- Basic usage with scikit-learn pipelines
- Cross-validation integration
- Grid search compatibility
- Custom transformer implementation
