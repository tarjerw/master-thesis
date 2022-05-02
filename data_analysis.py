import csv
from numpy.lib.histograms import histogram
import pandas as pd
import datetime
import calendar
import json

import numpy as np
from pytz import HOUR

import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

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
    #"Hour",
    "System Price",
    #"Month",
    #"Weekday",
    #"Season",
    #"Holiday",
    "Oslo",
    "Kr.sand",
    "Tr.heim",
    "Tromsø",
    "Bergen",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
    "DK1",
    "DK2",
    "FI",
    "Oil",
    "Coal",
    "Gas",
    "Low Carbon",
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
# print(selected_data[daily_data.Date == "2017-07-23"])  # get data for certain date


#print(hourly_data[selected_colums].describe())

def print_correlation_matrix(startdate="2014-01-01", enddate="2019-06-02"):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    correlationMatrix = selected_data[start_index:end_index].corr()
    h_map = sn.heatmap(correlationMatrix, annot=True, fmt='.2f', annot_kws={'size':7.5}, xticklabels=selected_data.columns, yticklabels=selected_data.columns)
    h_map.set_xticklabels(h_map.get_xticklabels(), fontsize=8, rotation=45, color='black')
    #h_map.set_yticklabels(h_map.get_yticklabels(), fontsize=8, color='black')
    plt.show()


#print_correlation_matrix()  # will print correlation matrix for all selected variables (selected_columns variable), also possible to set start/ end date


# a scatter plot comparing two variables (e.g., System Price and Oil)
def make_scatterplot(x, y, startdate="2014-01-01", enddate="2020-12-31"):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    sn.lmplot(
        x="System Price",
        y="Oil",
        data=hourly_data[start_index:end_index],
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
    x=["Oslo"],
    startdate="2014-01-01",
    enddate="2020-12-31",
    alpha=0.5,
    num_bins=50,
):
    start_index = date_list.index(startdate)
    end_index = date_list.index(enddate)
    histData = pd.DataFrame(hourly_data, columns=x)
    histData[start_index:end_index]
    histData.plot.hist(bins=num_bins, alpha=alpha, color='b', subplots=True, xticks=[x for x in range(-10, 100, 10)], xlim=(-10, 100), ylabel='')
    plt.show()


#get_histogram(x=['Oslo', 'Kr.sand', 'Tr.heim', 'Tromsø','Bergen'], enddate='2019-06-02') #Date has been modified to only include training data


def get_quantiles(division=0.1, var="System Price"):
    qunatile_data = pd.DataFrame(hourly_data, columns=[var])
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


def plot_large_system_price(training_data, selected_columns=['Date', 'Hour', 'Tr.heim']):
    data = training_data[selected_columns]
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(12, 7))

    #Dropping 29th of february 2016 - makes fuzz and is annoying
    data = data.drop(data[data['Date'] == '2016-02-29'].index)

    #Making a column for the date boject, and adjusting the hour thereafter
    data['Date_obj'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    for row in range(len(data)):
        data['Date_obj'].iloc[row] = data['Date_obj'].iloc[row].replace(hour = int(data['Hour'].iloc[row]))

    #making a seperate column for dates in a year - as an index named idx
    data['idx'] = data['Date'].str.slice(start=5, stop=10) + str(data['Hour'])

    #Dividing the data into the different years - enabling plotting of seperate years
    year_groups = {}
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    for year in years:
        year_groups[year] = data.loc[pd.DatetimeIndex(data['Date']).year==year]

    x_axis = year_groups[years[0]]['Date_obj']
    y_values = None
    for i in range(len(years)):
        y_values = year_groups[years[i]]['Tr.heim']
        axs[i].plot(x_axis, y_values, color='b')
        axs[i].set_ylim([-5, 75])
        axs[i].set_yticks(np.linspace(start=0,stop=75, num=6))
        axs[i].set_ylabel(str(years[i]))

    ax = plt.gca()
    days_major = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(days_major)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    #fig.legend("NO3", loc='upper center')

    plt.show()
    

    #major_locator = mdates.DayLocator(interval=30)

#plot_large_system_price(training_data)



def plot_avg_deviations(data, price_col, deviation_col='Holiday', save_dict=True):
    #data = data[[price_col, deviation_col]]
    avg = data[price_col].mean()
    #print(avg)
    deviation_col_mean = data.groupby(deviation_col).mean()
    #print(deviation_col_mean/avg) #Remove/add avg for monthly average price or coefficient
    coeff_df = deviation_col_mean/avg
    coeff_df = pd.DataFrame(coeff_df.iloc[1]/coeff_df.iloc[0])
    print(coeff_df)
    if save_dict:
        coeff_dict = {}
        for row in coeff_df.index:
            for col in coeff_df.columns:
                key = str(row)
                val = coeff_df[col].loc[row]
                coeff_dict[key] = val
        with open('data_erling/holiday_coefficients.json', 'w') as file:
            json_obj = json.dumps(coeff_dict)
            file.write(json_obj)
    
    #Used for plotting:
    '''  
    #deviation_col_mean.index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    #deviation_col_mean.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    deviation_col_mean.index = [x for x in range(0,24)]
    #deviation_col_mean.index = [x for x in range(0,2)]
    plt.bar(deviation_col_mean.index, deviation_col_mean[price_col], color='b')
    plt.xticks(color='black', fontsize=12, rotation=45)
    plt.xlabel('Month', color='black')
    plt.yticks(color='black', fontsize=15)
    plt.ylabel(str(price_col) +  ' area price deviation (€/MWh)', color='black')
    plt.show()
    '''


#dev_cols = ["Oslo", 'Kr.sand', 'Tr.heim', 'Tromsø','Bergen', 'SE1', 'SE2', 'SE3', 'SE4', 'DK1', 'DK2', 'FI'] #'Month', 'Weekday', 'Holiday'

#plot_avg_deviations(training_data[["Oslo", 'Kr.sand', 'Tr.heim', 'Tromsø','Bergen', 'SE1', 'SE2', 'SE3', 'SE4', 'DK1', 'DK2', 'FI', 'Holiday']], dev_cols)


def calc_quantiles(data, selected_columns=["Oslo", 'Kr.sand', 'Tr.heim', 'Tromsø','Bergen'], q_range=[0.01, 0.05, 0.1, 0.15, 0.5, 0.85, 0.9, 0.95, 0.99]):
    for col in selected_columns:
        string = [str(data[col].quantile(q)) + ' & ' for q in q_range]
        print(string)
    '''
    for q in q_range:
        print(str(q) + ' Quantile: ' + str(data[selected_columns].quantile(q)))
    print('max: ' + str(data[selected_columns].max()))
    print('min: ' + str(data[selected_columns].min()))
    '''

#calc_quantiles(training_data)

def find_skeweness(data, columns=['System Price']):
    print(data[columns].skew())

#find_skeweness(training_data, columns=selected_colums)

def find_kurtosis(data, columns=['System Price']):
    kurt = data[columns].kurtosis(axis=0)
    print(kurt)

#find_kurtosis(training_data, columns=selected_colums)

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

price_cols = [
    "Oslo",
    "Kr.sand",
    "Tr.heim",
    "Tromsø",
    "Bergen",
    #"SE1",
    #"SE2",
    #"SE3",
    #"SE4",
    #"DK1",
    #"DK2",
    #"FI"
    ]

def plot_correlation_plot(data, cols=price_cols, partial=False):
    fig, axs = plt.subplots(len(price_cols), sharex=True, figsize=(10,7))
    plt.ylim((0,1.5))
    cols = [x for x in cols if x in data.columns]
    if not partial:
        for col in range(len(cols)):
            plot_acf(data[price_cols[col]], ax=axs[col], title='', zero=False) #Is ugly and has too many titles
            axs[col].set_ylabel(cols[col], rotation=0, size=9, labelpad=20.0)
            axs[col].set_ylim((0, 1.0))
    else:
        for col in range(len(cols)):
            plot_pacf(data[price_cols[col]], ax=axs[col], title='', alpha=0.05, zero=False, lags=168) #Is ugly and has too many titles
            axs[col].set_ylabel(cols[col], rotation=0, size=9, labelpad=20.0)
            axs[col].set_ylim((0, 1.5))
    plt.show()

#plot_correlation_plot(training_data, partial=True)


def test_stationarity(data, cols=price_cols):
    cols = [x for x in cols if x in data.columns]
    print(adfuller(data[cols]))

#test_stationarity(training_data)

def test_for_autocorrelation(data, cols=price_cols):
    cols = [x for x in cols if x in data.columns]
    for col in cols:
        print(acorr_breusch_godfrey(data[col]))

test_for_autocorrelation(training_data)