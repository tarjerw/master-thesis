import csv
from numpy.lib.histograms import histogram
import pandas as pd
import datetime
import calendar

import numpy as np

import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator

from data_processing import training_test_split

plt.style.use("ggplot")

"""
TIME:
    "Date", (2014-01-01 -> , str)
    "Week", (1, int)
    "Month", (1, int)
    "Season", (1, int, 1-4)
    "Weekday", (3, int, 1-7)
    "Weekend", (0, int, 0-1)
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
    "Curve Demand", (884300.0)
"""


# retrieves data
data_path = "data_erling/hourly_data_areas.csv"
hourly_data = pd.read_csv(data_path)

date_list = hourly_data["Date"]
date_list = (
    date_list.values.tolist()
)  # list of all dates, can be used to get index from date, and vice versa

# dataframe with only selected columns
selected_colums = [
    #"Date",
    "System Price",
    #"Month",
    #"Weekday",
    #"Season",
    #"Oil",
    #"Coal",
    #"Gas",
    #"Low Carbon",
    "DK1",
    "DK2",
    "Oslo",
    "Kr.sand",
    "Bergen",
    "Molde",
    "Tr.heim",
    #"APX",
    #"OMEL",
    #"EEX",
    #"Total Vol",
    #"Supply",
    #"Demand",
    #"T Nor",
    #"Prec Norway",
    #"Prec Norway 7",
    #"Snow Norway",
    #"Wind SE",
    #"Wind Prod",
    #"Total Hydro",
    #"Total Hydro Dev",
]  # what vars to include

training_test_split = training_test_split  # data to split between training and test, 1979: last day of training June 2th 2019
selected_data = pd.DataFrame(hourly_data, columns=selected_colums)
training_data = selected_data[0:training_test_split]
test_data = selected_data[training_test_split:]


# DATA VISUALIZATION/ SUMMARY:

# print(daily_data["System Price"].describe()) #gives mean and std
# print(selected_data[daily_data.Date == "2017-07-23"])  # get data for certain date


def print_correlation_matrix(startdate="2014-01-01", enddate="2019-06-02"):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    correlationMatrix = selected_data[start_index:end_index].corr()
    h_map = sn.heatmap(correlationMatrix, annot=True, fmt='.2f', annot_kws={'size':9})
    h_map.set_xticklabels(h_map.get_xticklabels(), fontsize=8, rotation=45, color='black')
    h_map.set_yticklabels(h_map.get_yticklabels(), fontsize=8, color='black')
    plt.show()


print_correlation_matrix()  # will print correlation matrix for all selected variables (selected_columns variable), also possible to set start/ end date


# a scatter plot comparing two variables (e.g., System Price and Oil)
def make_scatterplot(x, y, startdate="2014-01-01", enddate="2020-12-31"):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    sn.lmplot(
        x="System Price",
        y="Oil",
        data=daily_data[start_index:end_index],
        fit_reg=True,
    )
    plt.show()


# make_scatterplot(x="System Price", y="Week")

# input list of vars, get graph of them or box plot
def get_graph(
    var_list=["System Price"],
    startdate="2014-01-01",
    enddate="2020-12-31",
    box_plot=False,
):
    graph_data = pd.read_csv(
        "external_data/all_data_daily.csv", index_col=0, parse_dates=True
    )
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    graph_data = pd.DataFrame(graph_data, columns=var_list)
    if box_plot:
        graph_data.plot.box()
    else:
        graph_data[start_index:end_index].plot()

    plt.show()


def get_histogram(
    x=["System Price"],
    startdate="2014-01-01",
    enddate="2020-12-31",
    alpha=0.5,
    num_bins=40,
):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    histData = pd.DataFrame(daily_data, columns=x)
    histData[start_index:end_index]
    histData.plot.hist(bins=num_bins, alpha=alpha, color='b')
    plt.show()


#get_histogram(x=["System Price"], enddate='2019-06-02') #Date has been modified to only include training data


def get_quantiles(division=0.1, var="System Price"):
    qunatile_data = pd.DataFrame(daily_data, columns=[var])
    quantile = division
    while quantile <= 1.0:
        num = qunatile_data.quantile(quantile)
        print(f"quantile {quantile}: {num}")
        quantile += division
    print(qunatile_data.describe())


# can also get boxplot
'''
get_graph(
    var_list=["System Price", "Oil", "Coal", "Gas"],
    startdate="2016-01-01",
    enddate="2017-01-31",
    box_plot=True,
)
'''

def add_nan_values(x_axis, y_axis, backpadding=True):
    values = np.zeros(x_axis.shape)
    values[:] = np.nan
    diff = len(x_axis) - len(y_axis)
    if backpadding:
        values[diff:] = y_axis
    else:
        values[:len(y_axis)] = y_axis
    return values



def plot_large_system_price(training_data, selected_columns=['Date', 'System Price']):
    data = training_data[selected_columns]
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(10, 7))
    data['Date_obj'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    #making a seperate column for dates in a year - as an index named idx
    data['idx'] = data['Date'].str.slice(start=5, stop=10)
    year_groups = []
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    #Putting all the data from the different years into seperate dataframes
    for i in range(len(years)):
        year_groups.append(data.loc[pd.DatetimeIndex(data['Date_obj']).year == years[i]])
    
    x_axis = year_groups[1]['idx']
    y_values = None
    for i in range(len(years)):
        if i == 0:
            y_values = add_nan_values(x_axis, year_groups[i]['System Price'].to_numpy())
        elif i == len(years) - 1:
            y_values = add_nan_values(x_axis, year_groups[i]['System Price'].to_numpy(), backpadding=False)
        else:
            y_values = year_groups[i]['System Price'].to_numpy()
            x_axis = year_groups[i]['idx']
        axs[i].plot(x_axis, y_values, color='blue')
        axs[i].set_ylim([0,85])
        axs[i].set_yticks(np.linspace(start=0,stop=80, num=5))
        axs[i].set_ylabel(str(years[i]))
    
    
    days_major = mdates.MonthLocator(interval=1)
    plt.gca().xaxis.set_major_locator(days_major)
    fig.legend(loc='upper center')

    plt.show()
    

    #major_locator = mdates.DayLocator(interval=30)

#plot_large_system_price(training_data)



def plot_avg_deviations(data, deviation_col='Month'):
    data = data[['System Price', deviation_col]]
    avg = np.average(data['System Price'])
    deviation_col_mean = data.groupby(deviation_col).mean() - avg
    print(deviation_col_mean/avg)
    deviation_col_mean.index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    #deviation_col_mean.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.bar(deviation_col_mean.index, deviation_col_mean['System Price'], color='b')
    plt.xticks(color='black', fontsize=15, rotation=45)
    plt.xlabel('Month', color='black')
    plt.yticks(color='black', fontsize=15)
    plt.ylabel('System Price Deviation (€/MWh)', color='black')
    plt.show()

#plot_avg_deviations(training_data)

def calc_quantiles(data, selected_columns=['System Price'], q_range=[0.01, 0.05, 0.1, 0.15, 0.5, 0.85, 0.9, 0.95, 0.99]):
    for q in q_range:
        print(str(q) + ' Quantile: ' + str(data[selected_columns].quantile(q)))
    print('max: ' + str(data[selected_columns].max()))
    print('min: ' + str(data[selected_columns].min()))

#calc_quantiles(training_data)

def find_skeweness(data, selected_columns=['System Price']):
    print(data[selected_columns].skew())

#find_skeweness(training_data)

def find_kurtosis(data, columns=['System Price']):
    kurt = data[columns].kurtosis(axis=0)
    print(kurt)

#find_kurtosis(training_data)

def find_vol_and_avg_for_yrs(data, columns=['System Price', 'Date', 'Season']):
    data['Date Obj']  = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['Year'] = pd.DatetimeIndex(data['Date Obj']).year
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    seasons = [1.0, 2.0, 3.0, 4.0]
    #for s in seasons:
    for y in years:
        print('CURRENT YEAR IS ----> ' + str(y))
        #print('CURRENT SEASON IS ----> ' + str(s)) 
        avg_sys_price = data.loc[(data['Year']==y)]['System Price'].mean()
        std = data.loc[(data['Year']==y)]['System Price'].std()
        print('AVERAGE SYSTEM PRICE -> ' + str(avg_sys_price))
        print('STANDARD DEVIATION -> ' + str(std))

#find_vol_and_avg_for_yrs(training_data)

