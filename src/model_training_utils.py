from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import json
import yaml
import os
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

def create_objective(X, y, model_class):
    def objective(trial):
        n_features = trial.suggest_int('n_features', 1, X.shape[1])
        params = {}
        if model_class == Ridge:
            params['alpha'] = trial.suggest_float('alpha', 1, 50, log=True)

        elif model_class == Lasso:
            params['alpha'] = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            
        elif model_class == RandomForestRegressor:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=25)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10, step=2)

        elif model_class == XGBRegressor:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.05)

        model = model_class(**params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("feature_selection", SelectKBest(score_func=f_regression, k=n_features)),
            ("regressor", model)
        ])

        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_root_mean_squared_error")
        return -scores.mean()
    
    return objective

def train_best_model(X_train, y_train):
    with open(parent_dir+"/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    with open(parent_dir+config['model']['params_path'], 'r') as f:
        optuna_best_results = json.load(f)

    best_params = optuna_best_results[config['model']['name']]
    n_features = best_params.pop("n_features")

    module = importlib.import_module(config['model']['module_name'])
    model = getattr(module, config['model']['name'])

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_regression, k=n_features)),
        ("regressor", model(**best_params))
    ])

    final_model.fit(X_train, y_train)
    return final_model