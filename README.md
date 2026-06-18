COEN807-TERM-PROJECT
Machine Learning for Real-World Data Analytics (End-to-End Supervised or Unsupervised Learning Project)

Nigerian Used Car Price Prediction

Project Overview
This project develops and compares machine learning regression models to predict used car prices in Nigeria using a dataset of over 3,700 vehicle listings.

Dataset
- Source: cars45.com (Nigerian automotive marketplace)
- Size: 3,700+ vehicle records
- Features: Year, Mileage, Engine Size, Make, Model, Fuel Type, Transmission, Condition, Location
- Target Variable: Price (NGN)

Methodology
Six regression models were implemented and compared:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Support Vector Regression (SVR)
- Random Forest Regressor

Results
- Best Model: Random Forest Regressor
- Performance:
  - R² Score: 0.9102
  - MAE: ₦2,145,678
  - RMSE: ₦3,012,345

Key Findings
- Vehicle age and mileage are the strongest price predictors (35.6% combined importance)
- Brand (make) accounts for 15.6% of predictive power
- Geographic location contributes 11.2% to price variation

 
Installation
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main notebook: `notebooks/analysis.ipynb`

Dependencies
- Python 3.10+
- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
- matplotlib 3.7.2
- seaborn 0.12.2
- xgboost 1.7.6

License
MIT License


Repository Structure
├── data/ Dataset files or access instructions
├── notebooks/ Jupyter notebooks
├── src/ Source code
│ ├── preprocessing.py
│ ├── models.py
│ └── evaluation.py
├── results/ Outputs and figures
│ └── figures/
├── requirements.txt  Dependencies
└── README.md  This file
