import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import seaborn as sns


import statsmodels.api as sm
from scipy import stats

from parameters import parameters

import statsmodels.stats.api as sms

selected_colums_regression = parameters["selected_colums_regression"]

output_variable = parameters["output_variable"]

if output_variable not in selected_colums_regression:
    selected_colums_regression.append(output_variable)
    print("added output variable into selected columns")

# retrieves data
hourly_data = pd.read_csv("data_erling/hourly_data_areas.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

# converting month, weekday and week to one-hot encoding
month_one_hot_encoding = False
if "Month" in selected_colums_regression:
    month_one_hot_encoding = True
    selected_colums_regression.remove("Month")  # want as seperate df
weekday_one_hot_encoding = False
if "Weekday" in selected_colums_regression:
    weekday_one_hot_encoding = True
    selected_colums_regression.remove("Weekday")  # want as seperate df
week_one_hot_encoding = False
if "Week" in selected_colums_regression:
    week_one_hot_encoding = True
    selected_colums_regression.remove("Week")  # want as seperate df
hour_one_hot_encoding = False
if "Hour" in selected_colums_regression:
    hour_one_hot_encoding = True
    selected_colums_regression.remove("Hour")  # want as seperate df

lin_reg_data = pd.DataFrame(hourly_data, columns=selected_colums_regression)
lin_reg_data = lin_reg_data.astype(float)

# converting month, weekday and week to one-hot encoding part 2
if month_one_hot_encoding:
    selected_colums_regression.append("Month")  # for documentation
    month_one_hot = pd.get_dummies(hourly_data["Month"], prefix="Month")
    lin_reg_data = pd.concat([lin_reg_data, month_one_hot], axis=1)
    del lin_reg_data['Month_1']
if weekday_one_hot_encoding:
    selected_colums_regression.append("Weekday")  # for documentation
    weekday_one_hot = pd.get_dummies(hourly_data["Weekday"], prefix="Weekday")
    lin_reg_data = pd.concat([lin_reg_data, weekday_one_hot], axis=1)
    del lin_reg_data['Weekday_1']
if week_one_hot_encoding:
    selected_colums_regression.append("Week")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Week"], prefix="Week")
    lin_reg_data = pd.concat([lin_reg_data, week_one_hot], axis=1)
    del lin_reg_data['Week_1']
if hour_one_hot_encoding:
    selected_colums_regression.append("Hour")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Hour"], prefix="Hour")
    lin_reg_data = pd.concat([lin_reg_data, week_one_hot], axis=1)
    del lin_reg_data['Hour_1']

training_test_split = date_hour_list.index(parameters["test_split"])
training_data_regression = lin_reg_data[0:training_test_split] # 2014-01-01-0 - 2019-12-31-23 # 6 years

# train Ordinary Least Squares model
def make_mlr_with_summary(days, training_data_regression,poly):
    x_sm = training_data_regression[:-24*days]
    y_sm = training_data_regression[output_variable][24*days:]

    polynomial_features= PolynomialFeatures(degree=poly)
    x_sm = polynomial_features.fit_transform(x_sm)

    y_sm = y_sm.reset_index()
    del y_sm["index"]
    X_train = sm.add_constant(x_sm)
    model = sm.OLS(y_sm, X_train)
    result = model.fit()

    print(result.summary())
    # get values of the residuals
    residual = result.resid
    # run tests and get the p values
    print('p value of Jarque-Bera test is: ', stats.jarque_bera(residual)[1])
    print('p value of Shapiro-Wilk test is: ', stats.shapiro(residual)[1])
    print('p value of Kolmogorov-Smirnov test is: ', stats.kstest(residual, 'norm')[1])
    print('p value of Breusch–Pagan test is: ', sms.het_breuschpagan(result.resid, result.model.exog)[1])
    print('p value of White test is: ', sms.het_white(result.resid, result.model.exog)[1])

#make_mlr_with_summary(1,training_data_regression,1)

def make_mlr(days, training_data_regression, output_variable, poly):
    x = training_data_regression[:-24*days]
    y = training_data_regression[output_variable][24*days:] 

    polynomial_features= PolynomialFeatures(degree=poly)
    x = polynomial_features.fit_transform(x)

    mlr = LinearRegression()  
    mlr.fit(x, y)
    return mlr

mlr_models = []
for i in range(1,31): # develop mlr models for different day horizions (1-31)
    mlr_models.append(make_mlr(i,training_data_regression,output_variable,2))
    #print("Intercept: ", mlr_models[0].intercept_)
    #print("Coefficients:")
    #print(list(zip(training_data_regression, mlr_models[0].coef_)))



def make_forecasts_regression(start_date, number_of_days, lin_reg_data,mlr_models, poly):
    forecast_start = date_hour_list.index(start_date) - 24
    forecast_basis_day = lin_reg_data[forecast_start:forecast_start+24] # day used as basis for forecast
    
    polynomial_features= PolynomialFeatures(degree=poly)
    forecast_basis_day = polynomial_features.fit_transform(forecast_basis_day)
    
    predictions = []
    for i in range(number_of_days):
        predictions.extend(mlr_models[i].predict(forecast_basis_day))
    return predictions
    
#forecasts = make_forecasts_regression('2020-08-02-5',10,lin_reg_data,mlr_models,2)
#print(forecasts)
