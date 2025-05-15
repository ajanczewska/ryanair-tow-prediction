import pandas as pd

def check_dates(df):
    df_copy = df.copy()
    df_copy['DepartureDate_converted'] = pd.to_datetime(df_copy['DepartureDate'], format='%d/%m/%Y')
    print('The data is chronological:', df_copy['DepartureDate_converted'].is_monotonic_increasing)

    mismatched = df_copy[
        (df_copy['DepartureDate_converted'].dt.day != df_copy['DepartureDay']) |
        (df_copy['DepartureDate_converted'].dt.month != df_copy['DepartureMonth']) |
        (df_copy['DepartureDate_converted'].dt.year != df_copy['DepartureYear'])
    ]
    if mismatched.empty:
        print("All records match between 'DepartureDate' and day/month/year columns.")
    else:
        print("Records with different dates in columns:")
        display(mismatched)

def check_route(df):
    route = df['Route'].str.split('-', expand=True)
    mismatched = df[
        (route[0] != df['DepartureAirport'])|
        (route[1] != df['ArrivalAirport'])
    ]
    if mismatched.empty:
        print("All records match between 'Route' and 'DepartureAirport', 'ArrivalAirport' columns.")
    else:
        print("Records with different dates in columns:")
        display(mismatched)

