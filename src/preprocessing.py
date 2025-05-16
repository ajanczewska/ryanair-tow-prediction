import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def clean_data(df, train_set=True):
    df.replace('(null)', np.nan, inplace=True)
    # sort data by date
    df = df.sort_values('DepartureDate').reset_index(drop=True)
    # replace values with none
    df.replace('(null)', np.nan, inplace=True)
    df.replace('(null)', np.nan, inplace=True)
    if train_set:
        # remove missing values
        df.dropna(axis=0, subset=['ActualTOW'], inplace=True)
        df['ActualTOW'] = pd.to_numeric(df['ActualTOW'])
    # change data to numeric
    for col in ['ActualFlightTime', 'ActualTotalFuel', 'FLownPassengers',
                'BagsCount', 'FlightBagsWeight']:
        df[col] = pd.to_numeric(df[col])
    
    # encoder categorical data
    for col in ['DepartureAirport',	'ArrivalAirport', 'Route']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.reset_index(inplace=True, drop=True)
    return df

def imput_missing_values(df, history_set=pd.DataFrame()):
    df_copy = df.copy()
    for i in tqdm(range(1, len(df))):
        row = df_copy.iloc[i].copy()
        history = pd.concat([history_set, df_copy.iloc[:i].copy()])

        pax_by_route = history.groupby('Route')['FLownPassengers'].median().dropna().round()
        pax_by_arrival = history.groupby('ArrivalAirport')['FLownPassengers'].median().dropna().round()
        pax_by_departure =  history.groupby('DepartureAirport')['FLownPassengers'].median().dropna().round()
        global_pax_median = history['FLownPassengers'].median().round()

        # input passengers count
        pax = row['FLownPassengers']
        if pd.isna(pax):
            pax = (
                pax_by_route.get(row['Route']) or
                pax_by_arrival.get(row['ArrivalAirport']) or
                pax_by_departure.get(row['DepartureAirport']) or
                global_pax_median
            )
        df_copy.loc[i, 'FLownPassengers'] = pax

        # input bags count
        bags_per_pax = (df_copy['BagsCount']/df_copy['FLownPassengers']).dropna().mean()
        bags = row['BagsCount']
        if pd.isna(bags):
            bags = round(pax*bags_per_pax)
        df_copy.loc[i, 'BagsCount'] = bags

        # input bags weight
        avg_bag_weight = (df_copy['FlightBagsWeight']/df_copy['BagsCount']).dropna().mean()
        bags_weight = row['FlightBagsWeight']
        if pd.isna(bags_weight):
            bags_weight = round(bags*avg_bag_weight)

        df_copy.loc[i, 'FlightBagsWeight'] = bags_weight
        
    return df_copy