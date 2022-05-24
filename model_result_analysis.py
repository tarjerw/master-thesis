import json 
import pickle

from resultpaths import area_order, SARIMA, ARIMA, DNN, LSTM, LSTM2, GRU, GRU2, ENTCN, Naive, Reg1, Reg2

selected_area = "DK1"
error_metric = "MAE"


error_metric = error_metric + "_list.json"
area_index = area_order.index(selected_area)

BASE_PATH = str(ARIMA[area_index])
path = BASE_PATH + "/" + error_metric
date_path = BASE_PATH + "/Date_list.json"

open_file = open(path, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

open_file = open(date_path, "rb")
date_list = pickle.load(open_file)
open_file.close()

print(loaded_list)
print(date_list)