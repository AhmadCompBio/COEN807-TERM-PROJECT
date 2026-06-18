"""
Data Preprocessing Module

This module contains functions for cleaning, encoding, and scaling
the car price dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    """Remove rows with missing values."""
    return df.dropna()

def handle_outliers(df, column, method='iqr'):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def feature_engineering(df):
    """Create new features."""
    from datetime import datetime
    df['Car_Age'] = datetime.now().year - df['Year']
    return df

def encode_categorical(df, columns, method='onehot'):
    """Encode categorical variables."""
    if method == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    else:
        return df  # placeholder for other encoding methods

def scale_features(df, columns, fit=False, scaler=None):
    """Scale numerical features."""
    if fit:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler
