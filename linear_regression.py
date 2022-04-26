import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import seaborn as sns


selected_colums = [
    "System Price",
    "Oslo",
    "Kr.sand",
    "Tr.heim",
    "Tromsø",
    "Bergen",
    "Month",
    "Weekday"
] 

output_variable = ( 
    "Oslo"  # what are we forecasting, in this thesis area bidding prices
)

if output_variable not in selected_colums:
    selected_colums.append(output_variable)
    print("added output variable into selected columns")

# retrieves data
hourly_data = pd.read_csv("data_erling/hourly_data_areas.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

# converting month, weekday and week to one-hot encoding
month_one_hot_encoding = False
if "Month" in selected_colums:
    month_one_hot_encoding = True
    selected_colums.remove("Month")  # want as seperate df
weekday_one_hot_encoding = False
if "Weekday" in selected_colums:
    weekday_one_hot_encoding = True
    selected_colums.remove("Weekday")  # want as seperate df
week_one_hot_encoding = False
if "Week" in selected_colums:
    week_one_hot_encoding = True
    selected_colums.remove("Week")  # want as seperate df
hour_one_hot_encoding = False
if "Hour" in selected_colums:
    hour_one_hot_encoding = True
    selected_colums.remove("Hour")  # want as seperate df

lin_reg_data = pd.DataFrame(hourly_data, columns=selected_colums)
lin_reg_data = lin_reg_data.astype(float)

# converting month, weekday and week to one-hot encoding part 2
if month_one_hot_encoding:
    selected_colums.append("Month")  # for documentation
    month_one_hot = pd.get_dummies(hourly_data["Month"], prefix="Month")
    lin_reg_data = pd.concat([lin_reg_data, month_one_hot], axis=1)
    del lin_reg_data['Month_1']
if weekday_one_hot_encoding:
    selected_colums.append("Weekday")  # for documentation
    weekday_one_hot = pd.get_dummies(hourly_data["Weekday"], prefix="Weekday")
    lin_reg_data = pd.concat([lin_reg_data, weekday_one_hot], axis=1)
    del lin_reg_data['Weekday_1']
if week_one_hot_encoding:
    selected_colums.append("Week")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Week"], prefix="Week")
    lin_reg_data = pd.concat([lin_reg_data, week_one_hot], axis=1)
    del lin_reg_data['Week_1']
if hour_one_hot_encoding:
    selected_colums.append("Hour")  # for documentation
    week_one_hot = pd.get_dummies(hourly_data["Hour"], prefix="Hour")
    lin_reg_data = pd.concat([lin_reg_data, week_one_hot], axis=1)
    del lin_reg_data['Hour_1']

training_test_split = date_hour_list.index('2020-01-01-0')
training_data = lin_reg_data[0:training_test_split] # 2014-01-01-0 - 2019-12-31-23 # 6 years


def make_mlr(days, training_data):
    x = training_data[:-24*days]
    y = training_data[output_variable][24*days:] 

    mlr = LinearRegression()  
    mlr.fit(x, y)
    return mlr

#mlr1 = make_mlr(1,training_data)
#print("Intercept: ", mlr1.intercept_)
#print("Coefficients:")
#print(list(zip(training_data, mlr1.coef_)))

mlr_models = []
for i in range(1,32): # develop mlr models for different day horizions (1-31)
    mlr_models.append(make_mlr(i,training_data))



def make_forecasts(start_date, number_of_days, lin_reg_data):
    forecast_start = date_hour_list.index(start_date)
    forecast_basis_day = lin_reg_data[forecast_start:forecast_start+24] # day used as basis for forecast
    
    predictions = []
    for i in range(number_of_days):
        predictions.extend(mlr_models[i].predict(forecast_basis_day))
    return predictions
    
forecasts = make_forecasts('2020-08-02-0',10,lin_reg_data)
