import pandas as pd
import numpy as np
from cgi import test
import json

selected_columns_naive = ["SE1", 
    "SE2", 
    "SE3", 
    "SE4", 
    "DK1", 
    "DK2", 
    "FI", 
    "Oslo", 
    "Kr.sand", 
    "Bergen", 
    "Tr.heim",
    "Tromsø",
    "System Price",
    "Weekday",
    "Holiday",
    "Month",
    "Hour"
] # should be no need to change this 

# retrieves data
hourly_data = pd.read_csv("data_erling/hourly_data_areas.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

naive_hourly_data = pd.DataFrame(hourly_data, columns=selected_columns_naive) #so to not include irrelevant columns

 
# Opening JSON file
with open('data_erling/coefficients/hour_coefficients.json') as json_file:
    hour_coefficients = json.load(json_file)
with open('data_erling/coefficients/month_coefficients.json') as json_file:
    month_coefficients = json.load(json_file)
with open('data_erling/coefficients/weekday_coefficients.json') as json_file:
    weekday_coefficients = json.load(json_file)
with open('data_erling/coefficients/holiday_coefficients.json') as json_file:
    holiday_coefficients = json.load(json_file)

def make_forecasts_naive(start_date, number_of_days,area,naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients, enhanced_naive = True, seven_day_lag = False):
    def get_dict_key(area,number): return "('" + str(area) + "', " + str(int(number)) + ")" # used to produce dict key for json

    forecast_start = date_hour_list.index(start_date) - 24
    forecast_basis_day = naive_hourly_data[forecast_start:forecast_start+24] # day used as basis for forecast
    forecast_basis_week = naive_hourly_data[forecast_start - 24*6:forecast_start+24]
    
    forecast_period = naive_hourly_data[forecast_start + 24:forecast_start+24 + 24*number_of_days]
    predictions = [] 

    if enhanced_naive == False: 
        if seven_day_lag:
            if number_of_days != 7:
                return "fuck you, number of days not equal to 7"
            for i in range(7*24):
                predictions.append(forecast_basis_week.iloc[i][area])
        else:
            for _ in range(number_of_days):
                for h in range(24):
                    predictions.append(forecast_basis_day.iloc[h][area])
        return predictions

    basis_value = sum(forecast_basis_day[area])/24 

    
        
    
    if forecast_period.iloc[0]["Holiday"] == 1:
        basis_value = basis_value * holiday_coefficients[area]
    basis_value = basis_value * weekday_coefficients[get_dict_key(area,forecast_basis_day.iloc[0]["Weekday"])] 
    
    current_month = forecast_basis_day.iloc[0]["Month"]
    if current_month == 1:
        prev_month = 12
        next_month = 2
    elif current_month == 12:
        prev_month = 11
        next_month = 1
    else:
        prev_month = current_month - 1
        next_month = current_month + 1

    #effect per day of monthly seasonality
    monthly_seasonality = (month_coefficients[get_dict_key(area,next_month)] / month_coefficients[get_dict_key(area,prev_month)])**(1/60)
    

    

    for i in range(number_of_days):
        forecast_day = forecast_period[i*24:(i+1)*24]
        day_coef = weekday_coefficients[get_dict_key(area,forecast_day.iloc[0]["Weekday"])]
        month_coef = monthly_seasonality ** (i+1)
        holiday_coef = 1
        if forecast_day.iloc[0]["Holiday"] == 1:
            holiday_coef = holiday_coefficients[area]

        for h in range(24):
            hour_coef = hour_coefficients[get_dict_key(area,forecast_day.iloc[h]["Hour"] + 1)]
            hour_prediction = basis_value * day_coef * holiday_coef * hour_coef
            predictions.append(hour_prediction)
    
    return predictions
#print(make_forecasts_naive("2017-02-04-0",5,"SE1",naive_hourly_data,hour_coefficients,month_coefficients,weekday_coefficients,holiday_coefficients,True))