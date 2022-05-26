
import json 
import pickle
from telnetlib import DM
import numpy as np
import scipy.stats as st

import seaborn as sn
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


from resultpaths import area_order, SARIMA, ARIMA, DNN, LSTM, LSTM2, GRU, GRU2, ENTCN, Naive, Reg1, Reg2

selected_area = "Oslo"
error_metric = "MAE"
model = "ENTCN"

result_path_dict = {
    "SARIMA" : SARIMA,
    "ARIMA": ARIMA,
    "DNN": DNN,
    "LSTM": LSTM,
    "S-LSTM":LSTM2,
    "GRU": GRU,
    "S-GRU":GRU2,
    "ENTCN":ENTCN,
    "Naive 7d":Naive,
    "Lin reg":Reg1,
    "Quad reg":Reg2
}

# Z-test intervals
# https://www.omnicalculator.com/statistics/critical-value to find more
print(st.norm.ppf(.975))
print(st.norm.cdf(-100))

open_file = open("models/tarjewang/Naive2-extra_Oslo_05.26.2022.21.08.57/forecast_dict.json", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


def get_error_list(model, area, error_metric = "MAE"):
    test_lenght = 339 
    area_index = area_order.index(area)
    BASE_PATH = str(result_path_dict[model][area_index])
    path = BASE_PATH + "/" + error_metric + "_list.json"
    date_path = BASE_PATH + "/Date_list.json"
    parameters_path = BASE_PATH + "/parameters.json"

    open_file = open(path, "rb")
    loaded_list = pickle.load(open_file)[-test_lenght:]
    open_file.close()

    open_file = open(date_path, "rb")
    date_list = pickle.load(open_file)[-test_lenght:]
    open_file.close()
    
    with open(parameters_path, 'r') as openfile:
        loaded_parameters = json.load(openfile)

    return loaded_list, date_list, loaded_parameters

_,da,_ =get_error_list(model,"Oslo")
print(da)





def DM_test(input_models, areas,region_name):
    models = []
    mod_per = []
    for mod in input_models:
        mod_loss = []
        for area in areas:
            mod1_in_area,_,_ = get_error_list(mod,area)
            mod_loss.extend(mod1_in_area)
        mod_per.append(np.average(mod_loss))
    
    while len(mod_per) > 0:
        min_val = min(mod_per)
        min_index = mod_per.index(min_val)
        models.append(input_models[min_index])
        input_models.pop(min_index)
        mod_per.pop(min_index)

    DM_matrix = [[0 for x in range(len(models))] for y in range(len(models))] 
    significance_matrix = [[1 for x in range(len(models))] for y in range(len(models))]
    for ind1 in range(len(models)):
        for ind2 in range(ind1+1,len(models)):
            model1 = models[ind1]
            model2 = models[ind2]
            model1_losses = []
            model2_losses = []
            for area in areas:
                mod1_in_area,_,_ = get_error_list(model1,area)
                mod2_in_area,_,_ = get_error_list(model2,area)
                model1_losses.extend(mod1_in_area)
                model2_losses.extend(mod2_in_area)
            delta_1_2 = []
            for ind, element in enumerate(model1_losses):
                delta_1_2.append(element - model2_losses[ind])
            delta_1_2 = np.array(delta_1_2)
            mean = np.mean(delta_1_2)
            std = np.std(delta_1_2)
            DM = abs(np.sqrt(len(delta_1_2))*(mean/std))
        
            significance_matrix[ind1][ind2] = round((1 - st.norm.cdf(DM))*2,4)
            DM_matrix[ind1][ind2] = round(abs(DM),2)
            
    print("      " +' '.join(map(str, models)))
    for ind,line in enumerate(DM_matrix):
        print (models[ind] + " " + ' '.join(map(str, line)))

    print("      " +' '.join(map(str, models)))
    for ind,line in enumerate(significance_matrix):
        print (models[ind] + " " + ' '.join(map(str, line)))

    print_correlation_matrix(significance_matrix,models,region_name)

    return DM_matrix,significance_matrix

def print_correlation_matrix(DM_matrix,models,region_name):
    mask = np.triu(np.ones_like(DM_matrix))
    DM_matrix = np.transpose(DM_matrix)

    cmap = sn.color_palette("coolwarm", as_cmap=True)
    h_map = sn.heatmap(DM_matrix, cmap = cmap,annot=True, fmt='.2f', annot_kws={'size':7.5}, xticklabels=models, yticklabels=models,mask=mask,vmin=0.01, vmax=0.1)
    h_map.set_xticklabels(h_map.get_xticklabels(), fontsize=8, rotation=45, color='black')
    h_map.set_title(region_name)
    plt.show()

def get_param_values(metric_string): # return list with metrics [mean, median, std, min, max]
    return [round(float(x.split(": ")[1]),3) for x in metric_string.split(",")]

def get_error_values(model, area, error_metric):
    _,_,params = get_error_list(model,area,error_metric)
    return get_param_values(params[error_metric])



models_used = [
    "SARIMA",
        "ARIMA",
        "DNN",
        "LSTM",
        "S-LSTM",
        "GRU",
        "S-GRU",
        "ENTCN",
        "Naive 7d",
        "Lin reg",
        "Quad reg"
    ]

area_loop =[["Oslo"],["Kr.sand"],["Tr.heim"],["Tromsø"],["Bergen"],["Oslo","Kr.sand","Tr.heim","Tromsø","Bergen"],["SE1","SE2","SE3","SE4"],["DK1","DK2"],["FI"],area_order]
title_loop = ["NO1","NO2","NO3","NO4","NO5","Norway","Sweden","Denmark","Finland","NordPool"]

'''
for ind, areas in enumerate(area_loop):
    models_used = [
    "SARIMA",
        "ARIMA",
        "DNN",
        "LSTM",
        "S-LSTM",
        "GRU",
        "S-GRU",
        "ENTCN",
        "Naive 7d",
        "Lin reg",
        "Quad reg"
    ]
    print(f"{areas} = {title_loop[ind]}")
    DM_test(models_used,areas,title_loop[ind]) 
'''




models_used = [
    "SARIMA",
        "ARIMA",
        "DNN",
        "LSTM",
        "S-LSTM",
        "GRU",
        "S-GRU",
        "ENTCN",
        "Naive 7d",
        "Lin reg",
        "Quad reg"
    ]

print("Model,Area,Metric,Mean,Median,Std,Min,Max,mod_ind,met_ind,area_ind")
for mod_ind,model in enumerate(models_used):
    for met_ind,metric in enumerate(["MAE","SMAPE","RMSE","MAPE"]):
        for area_ind,area in enumerate(["Oslo","Kr.sand","Tr.heim","Tromsø","Bergen","SE1","SE2","SE3","SE4","DK1","DK2","FI"]):
            save_str = model + "," + area + "," + metric 
            for val in get_error_values(model,area,metric):
                save_str = save_str + "," + str(val)
            save_str = save_str + "," + str(mod_ind)  + "," + str(met_ind)  + "," + str(area_ind)  
            print(save_str)