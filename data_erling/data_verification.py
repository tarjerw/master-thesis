#This file is made to ensure that the data which we has copied from the daily to the hourly granularity is correct

import pandas as pd

hourly_data = pd.read_csv('hourly_data_areas.csv')
daily_data = pd.read_csv('external_data/all_data_daily.csv')

daily_cols = ['Temp Hamar', 'Temp Krsand', 'Temp Namsos', 'Temp Troms',
              'Temp Bergen', 'Temp Norway', 'Supply', 'Demand', 'Oil',
              'Coal', 'Gas', 'Low Carbon', 'APX', 'OMEL', 'EEX', 'Wind SE',
              'Wind Prod', 'Prec Norway', 'Prec Norway 7', 'Snow Norway', 'Date']

print(daily_data[daily_cols].describe())

print(hourly_data[daily_cols].describe())