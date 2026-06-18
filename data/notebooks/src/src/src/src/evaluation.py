"""
Model Evaluation Module

This module contains functions for evaluating models and generating visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Calculate and return all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def plot_actual_vs_predicted(y_true, y_pred, title):
    """Create actual vs predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('Actual Price (NGN)')
    ax.set_ylabel('Predicted Price (NGN)')
    ax.set_title(f'{title}: Actual vs Predicted')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_residuals(y_true, y_pred, title):
    """Create residual analysis plots."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Residual vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Price (NGN)')
    axes[0].set_ylabel('Residuals (NGN)')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_xlabel('Residuals (NGN)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def create_comparison_table(results_dict):
    """Create performance comparison table from results."""
    df = pd.DataFrame({
        'Model': [],
        'R² Score': [],
        'MAE (₦)': [],
        'RMSE (₦)': [],
        'Training Time (s)': []
    })
    
    for model_name, results in results_dict.items():
        df = pd.concat([df, pd.DataFrame({
            'Model': [model_name],
            'R² Score': [f"{results['r2']:.4f}"],
            'MAE (₦)': [f"{results['mae']:,.2f}"],
            'RMSE (₦)': [f"{results['rmse']:,.2f}"],
            'Training Time (s)': [f"{results['train_time']:.2f}"]
        })], ignore_index=True)
    
    return df
