import csv
from numpy.lib.function_base import average
import pandas as pd
import datetime
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

"""
The test case study is designed by using data from 1 Jan 2014 to 31 Dec 2019 as training data and 1 Jan 2020 to
31 Dec 2020 as testing data. 

VARIABLES: 
# every class not spesified is a float. Everything except Date is float or int. Converted selected columns to floats

TIME:
    "Date", (2014-01-01, str)
    "Week", (1, int) (converted to one-hot encoding, but very long, leads to 53 input vars)
    "Month", (1, int) (converted to one-hot encoding)
    "Season", (1, int, 1-4) (rather use month as it incorporates it)
    "Weekday", (3, int, 1-7) (converted to one-hot encoding)
    "Weekend", (0, int, 0-1) (rather use weekday as it incorporates it)
    "Holiday", (1, int, 0-1)
    "Hour", (0-23, int) (converted to one-hot encoding)
PRICE DATA:
    "System Price", (28.25, EUR / MWh)
    "Oil", (81.12, eur per barrel crude oil)
    "Coal", (59.71)
    "Gas",(28.11)
    "Low Carbon", (100.32)
    "APX", (41.48)
    "OMEL", (12.86)
    "EEX", (17.38)
Regional Prices: (THESE ARE NEW!)
    "SE1", (EUR / MWh)
    "SE2", (EUR / MWh)
    "SE3", (EUR / MWh)
    "SE4", (EUR / MWh)
    "DK1", (EUR / MWh)
    "DK2", (EUR / MWh)
    "FI", (EUR / MWh)
    "Oslo", (EUR / MWh)
    "Kr.sand", (EUR / MWh)
    "Bergen", (EUR / MWh)
    "Molde", (EUR / MWh)
    "Tr.heim", (EUR / MWh)
    "Tromsø", (EUR / MWh)
MARKET DATA
    "Total Vol", (884300.0)
    "NO Buy Vol", (340886, int)
    "NO Sell Vol", (299489, int)
    "SE Buy Vol", (355291, int)
    "SE Sell Vol", (408669, int)
    "DK Buy Vol", (58534.4)
    "DK Sell Vol", (101167.0)
    "FI Buy Vol", (134389, int)
    "FI Sell Vol", (98827.4)
    "Nordic Buy Vol", (889099.0)
    "Nordic Sell Vol", (908152.0)
    "Baltic Buy Vol", (42944.1)
    "Baltic Sell Vol", (23891.8)
    "Supply", (1076510.0)
    "Demand", (1117970.0)
PRODUCTION DATA:
    "NO Hydro", (54821.7)
    "SE Hydro", (21822.1)
    "FI Hydro", (3736.57)
    "Total Hydro", (80380.4)
    "NO Hydro Dev", (-1803.51)
    "SE Hydro Dev", (1589.92)
    "FI Hydro Dev", (209.37)
    "Total Hydro Dev", (-4.24)
    "Wind DK", (46140, int)
    "Wind SE", (93454.6)
    "Wind Prod", (139595.0)
WEATHER DATA (only Norway):
    "T Nor", (4.4, celsius)
    "Temp Hamar", (temp in celsius, float)
    "Temp Krsand", (temp in celsius, float)
    "Temp Namsos", (temp in celsius, float)
    "Temp Troms", (temp in celsius, float)
    "Temp Bergen", (temp in celsius, float)
    "Temp Norway", (temp in celsius, float)
    "Prec Norway", (101.2)
    "Prec Norway 7", (981.4)
    "Snow Norway", (45.0983)
Marius & Jake Estimate:
    "Curve Demand", (884300.0) don't think we are using this
"""

# these are the vars which can be changed
selected_colums = [
    "Season",
    "Weekend",
    "Holiday",
    "System Price",
    "Total Vol",
    "Supply",
    "Demand",
    "Total Hydro",
    "Total Hydro Dev",
    "Wind Prod",
    "T Nor",
    "Prec Norway",
    "Prec Norway 7",
    "Snow Norway",
    "Month",
    "Weekday",
    "Hour",
    "Date"
] 

output_variable = ( 
    "Oslo"  # what are we forecasting, in this thesis "System Price"
)
if output_variable not in selected_colums:
    selected_colums.append(output_variable)

# Example format: date = 2014-01-01, hour = 3 (03:00-04:00)

#these need to be defined
input_length = 100
training_length = 10 * 24 # how many days to use in each training example
predict_change_delay = 7 # wtf to do with this? 
prediction_horizon = 10 # number of days looking forward

input_length = len(selected_colums)

# converting month, weekday and week to one-hot encoding
month_one_hot_encoding = False
if "Month" in selected_colums:
    month_one_hot_encoding = True
    input_length += 11
    selected_colums.remove("Month")  # want as seperate df
weekday_one_hot_encoding = False
if "Weekday" in selected_colums:
    weekday_one_hot_encoding = True
    input_length += 6
    selected_colums.remove("Weekday")  # want as seperate df
week_one_hot_encoding = False
if "Week" in selected_colums:
    week_one_hot_encoding = True
    input_length += 52
    selected_colums.remove("Week")  # want as seperate df
hour_one_hot_encoding = False
if "Hour" in selected_colums:
    week_one_hot_encoding = True
    input_length += 23
    selected_colums.remove("Hour")  # want as seperate df

# retrieves data
hourly_data = pd.read_csv("external_data/all_data_hourly.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

data_used = pd.DataFrame(hourly_data, columns=selected_colums)
#data_used = data_used.astype(float) Need to be added later, and then remove date from selected columns

# converting month, weekday and week to one-hot encoding part 2
if month_one_hot_encoding:
    selected_colums.append("Month")  # for documentation
    month_one_hot = pd.get_dummies(hourly_data["Month"], prefix="Month")
    data_used = pd.concat([data_used, month_one_hot], axis=1)
if weekday_one_hot_encoding:
    selected_colums.append("Weekday")  # for documentation
    weekday_one_hot = pd.get_dummies(hourly_data["Weekday"], prefix="Weekday")
    data_used = pd.concat([data_used, weekday_one_hot], axis=1)
if week_one_hot_encoding:
    selected_colums.append("Week")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Week"], prefix="Week")
    data_used = pd.concat([data_used, week_one_hot], axis=1)
if hour_one_hot_encoding:
    selected_colums.append("Hour")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Hour"], prefix="Hour")
    data_used = pd.concat([data_used, hour_one_hot_encoding], axis=1)

training_test_split = date_hour_list.index('2020-01-01-0')
training_data = data_used[0:training_test_split] # 2014-01-01-0 - 2019-12-31-23 # 6 years
test_data = data_used[training_test_split:] # 2020-01-01-0 - 2020-12-31-23 # 1 year , need to talk about COVID

# NEED TO DO SOME DATA PREPROCESSING HERE !!! 


# split into x (input vars) and y (target system price) data
training_data_x = training_data[0:-prediction_horizon * 24]
training_data_y = training_data[training_length :]
training_data_y = training_data_y[output_variable] # will get 2181 different training cases

test_data_x = test_data[0:-prediction_horizon * 24]
test_data_y = test_data[training_length :]
test_data_y = test_data_y[output_variable] # will get 356 different training cases


# convert training data DFs to np
training_data_x = training_data_x.to_numpy()
training_data_y = training_data_y.to_numpy()
test_data_x = test_data_x.to_numpy()
test_data_y = test_data_y.to_numpy()



training_x = [] # 2181 lists of input list of lenght training_lenght(10) * 24
training_y = [] # 2181 lists of output list of lenght prediction_horizon(10) * 24

test_x = [] # 356 lists of input list of lenght training_lenght(10) * 24
test_y = [] # 356 lists of output list of lenght prediction_horizon(10) * 24

ind = 0
while ind + training_length * 24  <= len(training_data_x):
    training_x.append(training_data_x[ind:ind+training_length*24])
    training_y.append(training_data_y[ind+training_length*24:ind+training_length*24+prediction_horizon*24])
    ind += 24 

ind = 0
while ind + training_length * 24  <= len(test_data_x):
    test_x.append(test_data_x[ind:ind+training_length*24])
    test_y.append(test_data_y[ind+training_length*24:ind+training_length*24+prediction_horizon*24])
    ind += 24 

print(test_y)