from asyncio import FastChildWatcher
from cgi import test
from cmath import inf
from random import triangular
from socketserver import ThreadingUnixDatagramServer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from error_metrics import MAE_error
import math


#Helping functions for SARIMA, but not actually connected to the model itself. Feel it is okay to have them outside the class, since these are not really class methods...

#Returns the error for a model for a given set of the (training) data - Can consider using other type of information criteria
def eval_model_for_grid_search(temp_mod):
    temp_mod = temp_mod.fit()
    return temp_mod.aic

#Search for optimal hyperparameters for the SARIMA-forecasting model
def grid_search_params(sarima_obj, training_data):
    candidate_model_params = {}
    for p in sarima_obj.p_params:
        for q in sarima_obj.q_params:
            for d in sarima_obj.d_params:
                for P in sarima_obj.P_params:
                    for Q in sarima_obj.Q_params:
                        for D in sarima_obj.D_params:
                            for m in sarima_obj.m_params:
                                #These are to avoid a model where the seasonal AR is the same as the model AR etc.
                                p_actual = p
                                q_actual = q
                                d_actual = d
                                '''
                                if p!= 0:
                                    if P%p==0:
                                        p_actual -= 1
                                if q != 0:
                                    if Q%q==0:
                                        q_actual -= 1
                                if d!=0:
                                    if D%d==0:
                                        d_actual -= 1
                                '''
                                order_temp = (p_actual, q_actual, d_actual)
                                seasonal_order_temp = (P, Q, D, m)
                                temp_mod = SARIMAX(training_data, order=order_temp, seasonal_order=seasonal_order_temp, enforce_stationarity=False) #enforce stationarity to avoid LU decomposition Error
                                score = eval_model_for_grid_search(temp_mod)
                                candidate_model_params[score] = (order_temp, seasonal_order_temp)
    top_score = min([x for x in candidate_model_params.keys() if (x != None and not math.isnan(x))])
    best_params = candidate_model_params[top_score]
    print("-------------------")
    print("Best parameters:")
    print(best_params)
    print("-------------------")
    return best_params



class SARIMA_model:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters):
        self.model = None
        self.order = parameters["SARIMA_order"]
        self.seasonal_order = parameters["SARIMA_seasonal_order"]
        self.grid_search = parameters["SARIMA_grid_search"]
        self.look_back = parameters["training_length"] *24  #Days as input
        #if we are doing the grid search, we must store the lists from which we extract the values
        if parameters["SARIMA_grid_search"]:
            self.p_params = parameters["SARIMA_p_params"]
            self.q_params = parameters["SARIMA_q_params"]
            self.d_params = parameters["SARIMA_d_params"]
            self.P_params = parameters["SARIMA_P_params"]
            self.Q_params = parameters["SARIMA_Q_params"]
            self.D_params = parameters["SARIMA_D_params"]
            self.m_params = parameters["SARIMA_m_params"]
            self.parameters_selected = False
        else:
            self.order = parameters["SARIMA_order"]
            self.seasonal_order = parameters["SARIMA_seasonal_order"]
            self.parameters_selected = True
        self.prediction_horizon = parameters["prediction_horizon"] * 24
        self.target_variable = parameters["output_variable"]
        self.thershold = parameters["SARIMA_threshold"]
        self.training_data = None
        return self


    def fit(self, features):        #, targets, batch_size, epochs, validation_split, shuffle, verbose):
        self.training_data = features[-(self.thershold + self.prediction_horizon): ]
        if self.parameters_selected:
            self.model = SARIMAX(self.training_data, order=self.order, seasonal_order=self.seasonal_order)
            self.model =  self.model.fit()
            ret = self
        else:
            if self.grid_search:
                #must perform a grid search here...
                training_data = self.training_data
                order, seasonal_order = grid_search_params(self, training_data)
                self.order = order
                self.seasonal_order = seasonal_order
                self.model = SARIMAX(training_data, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
                self.model = self.model.fit()
                ret = self
                self.parameters_selected = True
            else:
                self.model = SARIMAX(training_data, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
                self.model = self.model.fit()
                ret = self
                self.parameters_selected = True
        return ret


    #The variable test data is the hourly prices for the entire test period - in out case  - 1.1.2020 to 31.12.2020
    def predict(self, test_data):
        
        predictions = np.zeros((int((len(test_data) - (self.look_back + (self.prediction_horizon-24)))/24) ,self.prediction_horizon)) #Shape on the form [days, pred_horizon]
        for i in range(len(predictions)):
            pred_j = self.model.predict(start=1, end=self.prediction_horizon)
            predictions[i, :] = pred_j


            updated_variables = test_data[i*24: (i+1)*24] #extracting the next 24 variables - to be added as updated variables
            print(f"last! {test_data[(i+1)*24]}")
            self.model = self.model.append(updated_variables, refit=True)
        return predictions

    def save_model(self, path):
        self.model.save(path)