from __future__ import annotations
import enum

import tensorflow as tf

import numpy as np
from numpy.random import seed

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates


import datetime
import json 
import pickle

from keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Flatten,
    InputLayer,
    Conv1D,
    GlobalMaxPooling1D,
)
from keras.losses import KLDivergence
from keras.models import Sequential, load_model
from keras.models import Model
from keras import metrics
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.python.keras.backend import (
    binary_crossentropy,
    conv1d,
    categorical_crossentropy,
)
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.losses import BinaryCrossentropy

from CNN import CNN

from parameters import parameters  # here changes to be model are done!!

from LSTM import LSTM_model
from GRU import GRU_model
from sarima import SARIMA_model

# importing vars and funcs needed for regression and naive 
from linear_regression import lin_reg_data, make_mlr, make_forecasts_regression
from naive import naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients, make_forecasts_naive


from error_metrics import MAE_error, MAPE_error, RMSE_error, SMAPE_error, get_metrics

from data_processing import (
    input_length,
    date_hour_list,
    test_x,
    test_y,
    training_x,
    training_y,
    training_data_y, #Needed for SARIMA
    test_data
)

from tqdm import trange

# setting random seed for numpy and tensorflow XD
seed(parameters["numpy_random_seed"])
tf.random.set_seed(parameters["tenserflow_random_seed"])

def get_new_date(start_date,days_increment, min_1 = False):
    start_index = date_hour_list.index(start_date)
    new_index = start_index + days_increment * 24 - min_1
    return date_hour_list[new_index]


#initializing the chosen model
if parameters["model_used"] not in ["LSTM", "GRU", "SARIMA"]:
    # creating TCN/ CNN model based on parameters
    model = CNN().initialize(parameters,input_length)
elif parameters["model_used"] == "LSTM":
    model = LSTM_model().initialize(parameters)
elif parameters["model_used"] == "GRU":
    model = GRU_model().initialize(parameters)
elif parameters["model_used"] == "SARIMA":
    model = SARIMA_model().initialize(parameters)
else:
    print("Invalid model")


if parameters["model_used"] == "SARIMA":
    model = model.fit(training_data_y)
else: #history-variable will NOT work when using SARIMA
    history = model.fit(
    features=training_x,
    targets=training_y,
    batch_size=parameters["batch_size"],
    epochs=parameters["epochs"],
    validation_split=parameters["validation_split"],
    shuffle=False,
    verbose=parameters["verbose"],
    ) 


training_test_split = date_hour_list.index(parameters["test_split"])

training_data_regression = lin_reg_data[0:training_test_split]
mlr_models = []
for i in range(1,parameters["prediction_horizon"] + 1): # develop mlr models for different day horizions 
    mlr_models.append(make_mlr(i,training_data_regression,parameters["output_variable"],parameters["regression_poly"]))

if parameters["model_used"] == "SARIMA":
    CNN_forecasts = model.predict(test_data[parameters["output_variable"]])
else:
    CNN_forecasts = model.predict(test_x)



def run_test(model_used,start_time):
    date_time_index = date_hour_list.index(start_time)
    test_start = int((date_time_index - parameters["training_length"] * 24 - date_hour_list.index(parameters["test_split"]))/24)
    
    if test_start < 0:
        print("start_time not in test range")
    if start_time[-2:] != '-0':
        print("Error, test must start at hour 0, change start_time variable please")
        return
    
    #print("------------------")
    #print(test_y)
    #print(test_start)
    actual = test_y[test_start]
    
    

    regression_forecast = make_forecasts_regression(start_time,parameters["prediction_horizon"],lin_reg_data,mlr_models,parameters["regression_poly"])   
    naive_forecast = make_forecasts_naive(start_time,parameters["prediction_horizon"],parameters["output_variable"],naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients,parameters["enhanced_naive"])

    forecasted_values = []

    CNN_forecast = CNN_forecasts[test_start] #Had to remove this outside the if-else-structure, as the LSTM and GRU also access it

    if model_used == "Naive":
        forecasted_values = naive_forecast
    elif model_used == "Regression":
        forecasted_values = regression_forecast
    elif model_used == "TCN" or model_used =="DNN":
        CNN_forecast = CNN_forecasts[test_start] 
        if parameters["base_model"] == "naive":
            forecasted_values = (CNN_forecast * parameters["TCN_factor"]) + naive_forecast
        elif parameters["base_model"] == "regression":
            forecasted_values = (CNN_forecast * parameters["TCN_factor"]) + regression_forecast
        else:
            forecasted_values = CNN_forecast
    elif model_used=="LSTM" or model_used=="GRU":
        forecasted_values = CNN_forecast
    elif model_used=="SARIMA":
        forecasted_values = CNN_forecast
    else:
        print("NO MODEL SELECTED!! Change model_used varaible!")


    error_list = []
    # get list to calc error metrics:
    for ind, element in enumerate(forecasted_values):
        error_list.append([element,actual[ind]])

    print(start_time)
    print(get_metrics(error_list))

    return forecasted_values, actual, get_metrics(error_list), error_list

def run_complete_test(model_used, start_time, end_time = "none"):
    
    steps = len(CNN_forecasts)
    date = get_new_date(start_time,-1)

    forecast_dict = {}
    cummulative_error_list = []
    MAE_list = []
    RMSE_list = []
    SMAPE_list = []

    for i in range(steps):
        if date == end_time:
            break
        date = get_new_date(date,1)
        forecasted, actual, error_metrics, error_list = run_test(model_used,date)
        forecast_dict[date] = [forecasted,actual]
        MAE_list.append(error_metrics["MAE"])
        SMAPE_list.append(error_metrics["SMAPE"])
        RMSE_list.append(error_metrics["RMSE"])
        cummulative_error_list.extend(error_list)
         
    hour_error_list = [[]]
    for ind,element in enumerate(cummulative_error_list):
        i = ind % (24*parameters["prediction_horizon"])
        if len(hour_error_list) <= i:
            hour_error_list.append([element])
        else:
            hour_error_list[i].append(element)
    hour_error_MAE = [MAE_error(hour) for hour in hour_error_list]
    hour_error_SMAPE = [MAE_error(hour) for hour in hour_error_list]
    hour_error_RMSE = [MAE_error(hour) for hour in hour_error_list]
    print(f"hour error MAE: {hour_error_MAE}")
    print(f"hour error SMAPE: {hour_error_SMAPE}")
    print(f"hour error RMSE: {hour_error_RMSE}")

    print(get_metrics(cummulative_error_list))
    print(f"MAE; mean: {np.mean(MAE_list)}, median: {np.median(MAE_list)}, std: {np.std(MAE_list)}, min: {np.min(MAE_list)}, max: {np.max(MAE_list)}")
    print(f"SMAPE; mean: {np.mean(SMAPE_list)}, median: {np.median(SMAPE_list)}, std: {np.std(SMAPE_list)}, min: {np.min(SMAPE_list)}, max: {np.max(SMAPE_list)}")
    print(f"RMSE; mean: {np.mean(RMSE_list)}, median: {np.median(RMSE_list)}, std: {np.std(RMSE_list)}, min: {np.min(RMSE_list)}, max: {np.max(RMSE_list)}")
   
    return forecast_dict



run_complete_test(parameters["model_used"],get_new_date(parameters["test_split"],parameters["training_length"]))

def save_model(
    model, parameters
):  # used to save the anet model
    print("hey hey")
    BASE_PATH = f"models/{datetime.datetime.now().strftime('%m.%d.%Y/%H.%M.%S')}"
    model.save_model(BASE_PATH)
    parameters.pop("metrics")
    parameters.pop("verbose")

    with open(f"{BASE_PATH}/parameters.json", "w") as file:
        json.dump(parameters, file, indent=4)

save_model(model, parameters)


def visualize_date(model_used,date):
    forecasted, actual, error_metrics, error_list = run_test(model_used,date)
    plt.style.use(parameters["plt_style"]) 
    plt.plot(forecasted,label=f"Forecasted ")
    plt.plot(actual,label="Actual")
    plt.legend(loc="upper left")
    plt.title(f"Forecast ({model_used}) vs. actual {date} - {get_new_date(date,parameters['prediction_horizon'],True)}")
    plt.show()
print(test_x.shape)
print(test_y.shape)
#visualize_date(parameters["model_used"],"2020-02-02-0")