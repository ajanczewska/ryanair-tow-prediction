# Take-Off Weight Prediction

Regression model predicting Take-Off Weight of an aircraft.

## General info 

This project aims to build a regression model that predicts the **Take-Off Weight (TOW)** of an aircraft. The dataset includes Ryanair flights from 01.10.2016 to 15.10.2016. The model is evaluated using **Root Mean Squared Error (RMSE)** on a separate validation dataset.

## Project content description

- `notebooks/` - Jupyter notebooks for EDA and models training and comparing.
- `src/` - Python scripts with functions for preprocessing, creating visualizations, analysing data and model training.
- `results/` - Contains results as a CSV file.
- `report/` - Contains report with detailed description of the project and methodology.
- `main.py` - Main executable to run full training and prediction pipeline.
- `requirements.txt` - Required packages and dependencies.
- `config.yaml` - Configuration file with parameters and paths used across the project.

## Requirements

Install the dependencies using:

```
pip install -r requirements.txt
```

Main libraries used:

- pandas, numpy
- scikit-learn
- xgboost
- optuna
- matplotlib, seaborn, plotly-express
- pyyaml.

## How to run

To train the model and generate predictions for the validation dataset:

```
python main.py
```
The resulting predictions will be saved in `results/predictions.csv`.

## Methodology

1. **EDA**
Data overview, quality checking, missing values detection, analysis.

2. **Data Cleaning**
Handling missing values using route/airport-based medians and logical imputations.

3. **Feature Engineering**
Transformation of categorical variables and choosing feature based on analysis and importance.

4. **Model training**
Training multiple models:

    - Linear Regression

    - Ridge

    - Lasso

    - Random Forest Regressor

    - XGBoost Regressor

5. **Hyperparameter optimization**
Using Optuna with time-aware cross-validation.

6. **Evaluation**
Model selection based on RMSE score on a validation split.

## Notes

The project assumes that all features are known or estimated before the flight. This allows for predicting the take-off weight using all features in the dataset without assuming that any of them could lead to data leakage.

