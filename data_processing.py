from cProfile import label
from cgi import test
import csv
from re import A
import wave
from bleach import clean
from numpy.lib.function_base import average
import pandas as pd
import datetime
import numpy as np
from pyparsing import col
import seaborn as sn
import matplotlib.pyplot as plt

#Imports for preprocessing
#from data_erling.data_transforms.min_max_transform import Min_max_scaler
#from data_erling.data_transforms.standardize import Standardizer
from data_erling.data_transforms.column_types import time_series_columns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.decomposition import PCA
import pywt
from statsmodels.robust import mad

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
    "Temp Hamar", (temp in celsius, float) Oslo
    "Temp Krsand", (temp in celsius, float) Kr.Sand
    "Temp Namsos", (temp in celsius, float) Tr.heim
    "Temp Troms", (temp in celsius, float) Tromsø
    "Temp Bergen", (temp in celsius, float) Bergen
    "Temp Norway", (temp in celsius, float)
    "Prec Norway", (101.2)
    "Prec Norway 7", (981.4)
    "Snow Norway", (45.0983)
Marius & Jake Estimate:
    "Curve Demand", (884300.0) don't think we are using this
"""

# these are the vars which can be changed
selected_colums = [
    "SE1",
    #"SE2",
    #"SE3",
    #"SE4"
    #"DK1",
    #"DK2",
    #"FI",
    #"Oslo",
    "Kr.sand",
    "Tromsø"
    #"Bergen",
    #"Molde"
] 

output_variable = ( 
    "SE1"  # what are we forecasting, in this thesis "System Price"
)
if output_variable not in selected_colums:
    selected_colums.append(output_variable)
if "Date" in selected_colums:
    print("REMOVE DATE FROM SELECTED COLUMS/ OUTPUT VARIABLE, NOT FLOAT!")

#Data processing - normalization and standardization - Standard is standardization
standardize_data = True
min_max_normalize_data = False

if standardize_data and min_max_normalize_data:
    min_max_normalize_data = False

# Example format: date = 2014-01-01, hour = 3 (03:00-04:00)

#these need to be defined
training_length = 10  # how many days to use in each training example
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
hourly_data = pd.read_csv("data_erling/hourly_data_areas.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

data_used = pd.DataFrame(hourly_data, columns=selected_colums)
data_used = data_used.astype(float) #remove date from selected columns

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


def find_pca_explained_variance(pca_obj):
    plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))
    plt.show()


def principal_component_analysis(data, explained_variance=0.99):
    pca = PCA(n_components=explained_variance)
    pca = pca.fit(data)
    #find_pca_explained_variance(pca)
    return pca.transform(data), pca

    

training_data_trans, pca_obj = principal_component_analysis(training_data)

# NEED TO DO SOME DATA PREPROCESSING HERE !!! 

active_price_colums = [x for x in time_series_columns if x in training_data.columns]


def standardize_data_func(training_data, test_data, active_cols):
    pd.set_option("mode.chained_assignment", None) 
    training_length = len(training_data)
    test_length = len(test_data)
    for col in active_cols:
        scaler = StandardScaler()
        scaler.fit(training_data[col].to_numpy().reshape((-1, 1)))
        training_data[col] = scaler.transform(training_data[col].to_numpy().reshape((-1, 1))).reshape((training_length,))
        test_data[col] = scaler.transform(test_data[col].to_numpy().reshape((-1, 1))).reshape((test_length,))
    return pd.DataFrame(training_data, columns=training_data.columns), pd.DataFrame(test_data, columns=test_data.columns)


def min_max_normalize_data_func(training_data, test_data, active_cols):
    pd.set_option("mode.chained_assignment", None)
    training_length = len(training_data)
    test_length = len(test_data)
    for col in active_cols:
        scaler = MinMaxScaler()
        scaler.fit(training_data[col].to_numpy().reshape((-1, 1)))
        training_data[col] = scaler.transform(training_data[col].to_numpy().reshape((-1, 1))).reshape((training_length,))
        test_data[col] = scaler.transform(test_data[col].to_numpy().reshape((-1, 1))).reshape((test_length,))
    return pd.DataFrame(training_data, columns=training_data.columns), pd.DataFrame(test_data, columns=test_data.columns)

if standardize_data:
    training_data, test_data = standardize_data_func(training_data, test_data, active_price_colums)
elif min_max_normalize_data:
    training_data, test_data = min_max_normalize_data_func(training_data, test_data, active_price_colums)
else:
    print('No standardization/normalization was made to the data')


def unit_vector_normalization(data):
    return pd.DataFrame(normalize(data), columns=data.columns)



#Can only be used for training data - not sure if it is scientifically valid to use it for test data
#Unsure about functionality of this function, should consult a professor on NTNU
def wavelet_transform_data(data, mother_wavelet='coif1', n_levels=30, plot_transform=False):
    data_numpy = data.to_numpy()
    cols = data.columns
    steps = len(data)
    fig, axs = plt.subplots(len(cols))
    for i in range(len(cols)):
        time_series = data[cols[i]].to_numpy().reshape((steps,))
        coeffs = pywt.wavedec(time_series, mother_wavelet, level=n_levels)
        sigma = mad(coeffs[-1])/0.5
        thresh = sigma * np.sqrt(2 * np.log(steps))
        coeffs[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coeffs[1:])
        clean = pywt.waverec(coeffs[1:], mother_wavelet)
        if plot_transform:
            axs[i].plot(data_numpy[:, i], c='b')
            axs[i].plot(clean, c='r')
            axs[i].set_title(cols[i])
        data_numpy[:, i] = clean
    if plot_transform:
        plt.show()
    return pd.DataFrame(data_numpy, columns=cols)



def asinh_transform_dataset(training_data, test_data):
    cols = training_data.columns
    training_data = training_data.to_numpy()
    test_data = test_data.to_numpy()
    for row in range(len(training_data)):
        training_data[row, :] = np.arcsinh(training_data[row, :])   #Numpy is a lot faster here than tensorflow
    for row in range(len(test_data)):
        test_data[row, :] = np.arcsinh(test_data[row, :])
    return pd.DataFrame(training_data, columns=cols), pd.DataFrame(test_data, columns=cols)


# split into x (input vars) and y (target system price) data
training_data_x = training_data[0:-prediction_horizon * 24]
training_data_y = training_data[training_length * 24:]
training_data_y = training_data_y[output_variable] # will get 2181 different training cases

test_data_x = test_data[0:-prediction_horizon * 24]
test_data_y = test_data[training_length * 24 :]
test_data_y = test_data_y[output_variable] # will get 356 different training cases


# convert training data DFs to np
training_data_x = training_data_x.to_numpy()
training_data_y = training_data_y.to_numpy()
test_data_x = test_data_x.to_numpy()
test_data_y = test_data_y.to_numpy()




training_x = [] # 2181 - (training_lenght + prediction_horizon - 1) lists of input list of lenght training_lenght(10) * 24
training_y = [] # 2181 - (training_lenght + prediction_horizon - 1) lists of output list of lenght prediction_horizon(10) * 24

test_x = [] # 356 - (training_lenght + prediction_horizon - 1) lists of input list of lenght training_lenght(10) * 24
test_y = [] # 356 - (training_lenght + prediction_horizon - 1) lists of output list of lenght prediction_horizon(10) * 24

ind = 0
while ind + (training_length * 24)   <= len(training_data_x):
    training_x.append(training_data_x[ind:ind+training_length*24])
    training_y.append(training_data_y[ind:ind+prediction_horizon*24])
    ind += 24 

ind = 0
while ind + ( training_length * 24 ) <= len(test_data_x):
    test_x.append(test_data_x[ind:ind+training_length*24])
    test_y.append(test_data_y[ind:ind+prediction_horizon*24])
    ind += 24 

training_x = np.array(training_x)
training_y = np.array(training_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


def describe_data(data):
    print(data.mad())
    print(data.median())
    print(data.describe())

wrong_data = hourly_data[hourly_data['Oslo'].isnull()]

print(wrong_data)