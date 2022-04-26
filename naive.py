import pandas as pd
import numpy as np
from cgi import test
import json

selected_columns = ["SE1", 
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
]



# retrieves data
hourly_data = pd.read_csv("data_erling/hourly_data_areas.csv")

date_hour_list = hourly_data["Date"].values.tolist() # e.g., 2020-06-05-7 -> 07:00-08:00 on 5 June 2020
for index, element in enumerate(date_hour_list):
    hour = index % 24
    date_hour_list[index] = element + "-" + str(hour)

naive_hourly_data = pd.DataFrame(hourly_data, columns=selected_columns) #so to not include irrelevant columns

 
# Opening JSON file
def get_dict_key(area,number): return "('" + str(area) + "', " + str(number) + ")" # used to produce dict key for json
with open('data_erling/coefficients/hour_coefficients.json') as json_file:
    hour_coefficients = json.load(json_file)
with open('data_erling/coefficients/month_coefficients.json') as json_file:
    month_coefficients = json.load(json_file)
with open('data_erling/coefficients/weekday_coefficients.json') as json_file:
    weekday_coefficients = json.load(json_file)
with open('data_erling/coefficients/holiday_coefficients.json') as json_file:
    holiday_coefficients = json.load(json_file)

print(holiday_coefficients["Oslo"])
print(weekday_coefficients[get_dict_key("Oslo",1)])
print(month_coefficients[get_dict_key("Oslo",1)])
print(weekday_coefficients[get_dict_key("Oslo",1)])




#future_day_weekday = daily_data.iloc[day_index]["Weekday"]
#current_weekday = future_day_weekday

def make_forecasts(start_date, number_of_days):
    forecast_start = date_hour_list.index(start_date)
    
    
    predictions = []
    for i in range(number_of_days):
        predictions.extend(mlr_models[i].predict(forecast_basis_day))
    return predictions