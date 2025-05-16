from src.preprocessing import clean_data, imput_missing_values, encode_categorical
from src.model_training_utils import train_best_model
import pandas as pd
import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "config.yaml")

with open(file_path, "r") as f:
    config = yaml.safe_load(f)

def main():
    # read data 
    df_train = pd.read_csv(script_dir+config['data']['train_path'], sep='\t')
    df_val_raw = pd.read_csv(script_dir+config['data']['test_path'], sep='\t')
    df_val_raw['id'] = df_val_raw.index

    # clean and complete data
    print('Cleansing data and filling missing values...')
    df_train = clean_data(df_train)
    df_train = imput_missing_values(df_train)

    df_val = clean_data(df_val_raw, train_set=False)
    df_val = imput_missing_values(df_val, df_train)

    # features
    X_train = df_train[config['columns']['features']]
    y_train = df_train[config['columns']['target']]
    X_val = df_val[config['columns']['features']]
    X_train, X_val = encode_categorical(X_train, X_val, col='Route')

    # model training
    model = train_best_model(X_train, y_train)
    y_pred = model.predict(X_val)

    # save predictions
    results = df_val.copy()
    results[config['columns']['target']]= y_pred
    
    results_df = pd.merge(df_val_raw, results[[config['columns']['target'], 'id']], on='id')
    results_df = results_df.drop(columns='id')

    results_df.to_csv(script_dir+config['data']['output_path'], index=False, sep='\t')
    print(f"Predictions saved successfully in {config['data']['output_path']}!")

if __name__ == "__main__":
    main()

