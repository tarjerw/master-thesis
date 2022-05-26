
import json 
import pickle
import numpy as np
import scipy.stats as st

from resultpaths import area_order, SARIMA, ARIMA, DNN, LSTM, LSTM2, GRU, GRU2, ENTCN, Naive, Reg1, Reg2

selected_area = "Oslo"
error_metric = "MAE"
model = "ENTCN"

result_path_dict = {
    "SARIMA" : SARIMA,
    "ARIMA": ARIMA,
    "DNN": DNN,
    "LSTM": LSTM,
    "LSTM2":LSTM2,
    "GRU": GRU,
    "GRU2":GRU2,
    "ENTCN":ENTCN,
    "Naive":Naive,
    "Reg1":Reg1,
    "Reg2":Reg2
}

# Z-test intervals
# https://www.omnicalculator.com/statistics/critical-value to find more
print(st.norm.ppf(.975))
print(st.norm.cdf(-100))

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

def DM_test(models, areas):

    
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
            DM = np.sqrt(len(delta_1_2))*(mean/std)
            print("here")
            print(DM)
            print(st.norm.cdf(DM))

            significance_matrix[ind1][ind2] = round((1 - st.norm.cdf(DM))*2,4)
            DM_matrix[ind1][ind2] = round(DM,3)
            
    print("      " +' '.join(map(str, models)))
    for ind,line in enumerate(DM_matrix):
        print (models[ind] + " " + ' '.join(map(str, line)))

    print("      " +' '.join(map(str, models)))
    for ind,line in enumerate(significance_matrix):
        print (models[ind] + " " + ' '.join(map(str, line)))

    return DM_matrix,significance_matrix

DM_test(["LSTM", "LSTM2", "GRU", "GRU2", "DNN"],["Tr.heim"]) 


def get_param_values(metric_string): # return list with metrics [mean, median, std, min, max]
    return [round(float(x.split(": ")[1]),3) for x in metric_string.split(",")]

def get_error_values(model, area, error_metric):
    _,_,params = get_error_list(model,area,error_metric)
    return get_param_values(params[error_metric])



print(get_error_values("LSTM","Oslo","MAE"))