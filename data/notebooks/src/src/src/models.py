"""
Model Training Module

This module contains functions for training and tuning ML models.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time

def get_models():
    """Return dictionary of models and hyperparameter grids."""
    return {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {'alpha': [0.01, 0.1, 1, 10, 100]}
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'params': {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {}  # Limited due to computational cost
        }
    }

def train_with_tuning(X_train, y_train, model_config):
    """Train model with hyperparameter tuning using grid search."""
    model = model_config['model']
    params = model_config['params']
    
    if params:
        start_time = time.time()
        grid_search = GridSearchCV(
            model, params, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        return grid_search.best_estimator_, grid_search.best_params_, train_time
    else:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        return model, "None", train_time
