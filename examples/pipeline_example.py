"""
Example of using PyGRidge with scikit-learn Pipeline
==================================================

This example demonstrates how to use PyGRidge estimators within
scikit-learn's Pipeline and cross-validation framework.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

# Import PyGRidge components
from pygridge.src.blockridge import BlockRidge
from pygridge.src.group_lasso import GroupLasso

# Generate sample regression data
X, y = make_regression(n_samples=100, n_features=20, random_state=42)

# Create feature groups (example: group features by pairs)
groups = np.repeat(np.arange(10), 2)

# Create a pipeline with standardization and BlockRidge
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('block_ridge', BlockRidge(alpha=1.0))
])

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Example with GroupLasso
group_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('group_lasso', GroupLasso(groups=groups, alpha=1.0))
])

# Perform cross-validation with GroupLasso
group_cv_scores = cross_val_score(group_pipeline, X, y, cv=5)
print("\nGroupLasso Results:")
print(f"Cross-validation scores: {group_cv_scores}")
print(f"Mean CV score: {group_cv_scores.mean():.3f} (+/- {group_cv_scores.std() * 2:.3f})")

# Example of parameter tuning with GridSearchCV
if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid
    param_grid = {
        'block_ridge__alpha': np.logspace(-3, 3, 7)
    }
    
    # Create grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X, y)
    
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
