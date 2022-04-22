import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import seaborn as sns
from data_processing import training_data



print(training_data)

x = training_data[0:-25]
y = training_data['SE1'][25:]

mlr = LinearRegression()  
mlr.fit(x, y)
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))