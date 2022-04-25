import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import seaborn as sns
from data_processing import training_data


def make_mlr(days, data):
    if "Month_1" in data:
        del data['Month_1']
    if "Weekday_1" in data:
        del data['Weekday_1']
    if "Hour_1" in data:
        del data['Hour_1']
    if "Week_1" in data:
        del data['Week_1']
    x = data[:-24*days]
    y = data['Oslo'][24*days:] 

    mlr = LinearRegression()  
    mlr.fit(x, y)
    return mlr

mlr1 = make_mlr(2,training_data)
print("Intercept: ", mlr1.intercept_)
print("Coefficients:")
print(list(zip(training_data, mlr1.coef_)))