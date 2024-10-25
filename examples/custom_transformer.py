"""
Custom Transformer Example
========================

This example shows how to create custom scikit-learn compatible
transformers using PyGRidge components.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class GroupFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer that selects features based on group importance.
    
    This transformer uses PyGRidge's group lasso implementation to perform
    feature selection based on group structure.
    """
    
    def __init__(self, groups, alpha=1.0, threshold=1e-5):
        self.groups = groups
        self.alpha = alpha
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """Fit the transformer by identifying important groups."""
        # Validate input
        X = check_array(X)
        
        # Import here to avoid circular imports
        from pygridge.src.group_lasso import GroupLasso
        
        # Fit group lasso model
        self.group_lasso_ = GroupLasso(
            groups=self.groups,
            alpha=self.alpha
        )
        
        if y is not None:
            self.group_lasso_.fit(X, y)
        else:
            # For unsupervised case, use dummy target
            dummy_y = np.zeros(X.shape[0])
            self.group_lasso_.fit(X, dummy_y)
        
        # Identify important groups
        coef_norms = np.zeros(len(np.unique(self.groups)))
        for idx, group in enumerate(np.unique(self.groups)):
            group_mask = self.groups == group
            coef_group = self.group_lasso_.coef_[group_mask]
            coef_norms[idx] = np.linalg.norm(coef_group)
        
        self.selected_groups_ = coef_norms > self.threshold
        return self
    
    def transform(self, X):
        """Transform X by selecting features from important groups."""
        check_is_fitted(self)
        X = check_array(X)
        
        # Create mask for selected features
        feature_mask = np.zeros(len(self.groups), dtype=bool)
        for group_idx, selected in enumerate(self.selected_groups_):
            if selected:
                feature_mask |= (self.groups == group_idx)
        
        return X[:, feature_mask]

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    
    # Create groups (example: group features by pairs)
    groups = np.repeat(np.arange(10), 2)
    
    # Create pipeline with custom transformer
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('group_selector', GroupFeatureSelector(groups=groups, alpha=1.0)),
    ])
    
    # Fit and transform data
    X_transformed = pipeline.fit_transform(X, y)
    
    print("Original feature shape:", X.shape)
    print("Transformed feature shape:", X_transformed.shape)
