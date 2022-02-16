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
] 

output_variable = ( 
    "System Price"  # what are we forecasting, in this thesis "System Price"
)

# retrieves data
hourly_data = pd.read_csv("external_data/all_data_hourly.csv")

# Example format: date = 2014-01-01, hour = 3 (03:00-04:00)

#these need to be defined
input_length = 100
training_length = 100
predict_change_delay = 100
