from cProfile import label
from cgi import test
import csv
from re import A
import wave
#from bleach import clean
from numpy.lib.function_base import average
import pandas as pd
import datetime
import numpy as np
from pyparsing import col
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# importing vars and funcs needed for regression and naive 
from linear_regression import lin_reg_data, make_mlr, make_forecasts_regression
from naive import naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients, make_forecasts_naive

#Imports for preprocessing
#from data_erling.data_transforms.min_max_transform import Min_max_scaler
#from data_erling.data_transforms.standardize import Standardizer
from data_erling.data_transforms.column_types import time_series_columns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.decomposition import PCA
#import pywt
from statsmodels.robust import mad

from parameters import parameters

# these are the vars which can be changed
selected_colums = parameters["selected_colums"]

output_variable = parameters["output_variable"]
if output_variable not in selected_colums:
    selected_colums.append(output_variable)
if "Date" in selected_colums:
    print("REMOVE DATE FROM SELECTED COLUMS/ OUTPUT VARIABLE, NOT FLOAT!")

# Set base model which will
base_model = parameters["base_model"] # "naive", "regression"
regression_poly = parameters["regression_poly"] # what factor of polynomials in regression (1 = linear)
enhanced_naive = parameters["enhanced_naive"] 

#Data processing - normalization and standardization - Standard is standardization
standardize_data = parameters["standardize_data"] 
min_max_normalize_data = parameters["min_max_normalize_data"]

if standardize_data and min_max_normalize_data:
    min_max_normalize_data = False

# Example format: date = 2014-01-01, hour = 3 (03:00-04:00)

#these need to be defined
training_length = parameters["training_length"]  # how many days to use in each training example
prediction_horizon = parameters["prediction_horizon"] # number of days looking forward
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
    hour_one_hot_encoding = True
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
    data_used = pd.concat([data_used, week_one_hot], axis=1)

training_test_split = date_hour_list.index(parameters["test_split"])
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

active_price_colums = [x for x in time_series_columns if x in training_data.columns]

def standardize_data_func(training_data, test_data, active_cols): # avg = 0, sd = 1 
    pd.set_option("mode.chained_assignment", None) 
    training_length = len(training_data)
    test_length = len(test_data)
    for col in active_cols:
        scaler = StandardScaler()
        scaler.fit(training_data[col].to_numpy().reshape((-1, 1)))
        training_data[col] = scaler.transform(training_data[col].to_numpy().reshape((-1, 1))).reshape((training_length,))
        test_data[col] = scaler.transform(test_data[col].to_numpy().reshape((-1, 1))).reshape((test_length,))
    return pd.DataFrame(training_data, columns=training_data.columns), pd.DataFrame(test_data, columns=test_data.columns)


def min_max_normalize_data_func(training_data, test_data, active_cols): # min max 0 to 1
    pd.set_option("mode.chained_assignment", None)
    training_length = len(training_data)
    test_length = len(test_data)
    for col in active_cols:
        scaler = MinMaxScaler()
        scaler.fit(training_data[col].to_numpy().reshape((-1, 1)))
        training_data[col] = scaler.transform(training_data[col].to_numpy().reshape((-1, 1))).reshape((training_length,))
        test_data[col] = scaler.transform(test_data[col].to_numpy().reshape((-1, 1))).reshape((test_length,))
    return pd.DataFrame(training_data, columns=training_data.columns), pd.DataFrame(test_data, columns=test_data.columns)

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
test_data_y = test_data[training_length * 24 :]
test_data_y = test_data_y[output_variable] # will get 356 different training cases
training_data_y = training_data[training_length * 24:]
training_data_y = training_data_y[output_variable] # will get 2181 different training cases

# Do standardization
if standardize_data:
    training_data, test_data = standardize_data_func(training_data, test_data, active_price_colums)
elif min_max_normalize_data:
    training_data, test_data = min_max_normalize_data_func(training_data, test_data, active_price_colums)
else:
    print('No standardization/normalization was made to the data')

# split into x (input vars) and y (target system price) data
training_data_x = training_data[0:-prediction_horizon * 24]
test_data_x = test_data[0:-prediction_horizon * 24]


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


# developing regression models
training_data_regression = lin_reg_data[0:training_test_split] # 2014-01-01-0 - 2019-12-31-23 # 6 years
mlr_models = []
if base_model == "regression":
    for i in range(1,prediction_horizon + 1): # develop mlr models for different day horizions 
        mlr_models.append(make_mlr(i,training_data_regression,output_variable,regression_poly))

while ind + (training_length * 24)   <= len(training_data_x):
    forecast_start = date_hour_list[ind+training_length*24]

    training_x.append(training_data_x[ind:ind+training_length*24])
     
    # need to substract base model forecast here! Which will be added when forecasting in test!! 
    if base_model == "naive":
        base_model_forecast = make_forecasts_naive(forecast_start,prediction_horizon,output_variable,naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients,enhanced_naive)
    elif base_model == "regression":
        base_model_forecast = make_forecasts_regression(forecast_start,prediction_horizon,lin_reg_data,mlr_models,regression_poly)   
    else: # no model used
        base_model_forecast = [0] * prediction_horizon*24

    y = training_data_y[ind:ind+prediction_horizon*24] - base_model_forecast
    training_y.append(y)
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

describe_data(hourly_data["Kr.sand"])
describe_data(hourly_data["Oslo"])

print("all good :)")