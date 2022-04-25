import pandas as pd

daily_prices = pd.read_csv('all_data_daily.csv')
hourly_prices = pd.read_csv('all_data_hourly.csv')

print('Variables from daily data')
print([x for x in daily_prices.columns])


print('Joint variables')
print([x for x in daily_prices.columns if x in hourly_prices.columns])

print('Variables distinct for daily data')
print([x for x in daily_prices if x not in hourly_prices.columns])
