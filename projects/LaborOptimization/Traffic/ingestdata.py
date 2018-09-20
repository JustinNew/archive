# This is the module to read in Data.

import pandas as pd
import numpy as np
import os

def ReadInTrafficData(file):

    traffic = pd.read_csv(file)

    # Convert String to Datetime and back to string "2017-01-01"
    traffic['TRFFC_DTE'] = pd.to_datetime(traffic.TRFFC_DTE, infer_datetime_format=True)
    traffic['Date'] = traffic['TRFFC_DTE'].dt.strftime('%Y-%m-%d')

    # Sort the Traffic data by store, date and hour.
    traffic.sort_values(by=['STR_ID', 'Date', 'SALEHOUR'], inplace=True)

    # Get the Daily Traffic
    traffic_daily = traffic[['Date', 'STR_ID', 'VSTR_IN_CNT',
        'VSTR_OUT_CNT']].groupby(['STR_ID', 'Date']).agg({'VSTR_IN_CNT': np.sum, 'VSTR_OUT_CNT': np.sum}).reset_index()

    # Split the daily traffic by store and save into every store.
    store_list = traffic_daily.STR_ID.unique().tolist()

    # Check whether the directory exists, if not create the directory.
    if not os.path.exists('./StoreDailyTraffic'):
        os.mkdir('./StoreDailyTraffic')

    for store_id in store_list:
        temp = traffic_daily[traffic_daily['STR_ID'] == store_id].copy()
        temp.to_csv('./StoreDailyTraffic/DailyTraffic_' + str(store_id) + '.csv')

    return

if __name__ == '__main__':

    file = './Hourly_Traffic Beg 17 - Aug 18.txt'

    ReadInTrafficData(file)

    return