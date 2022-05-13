from asyncio import FastChildWatcher
from cgi import test
from cmath import inf
from random import triangular
from socketserver import ThreadingUnixDatagramServer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from error_metrics import MAE_error


#Helping functions for SARIMA, but not actually connected to the model itself. Feel it is okay to have them outside the class, since these are not really class methods...

#Returns the error for a model for a given set of the (training) data
def eval_model_for_grid_search(temp_mod, validation_data):
    temp_mod = temp_mod.fit()
    forecasts = np.zeros((len(validation_data), 2))
    forecasts[:, 1] = validation_data
    results = temp_mod.forecast(steps=len(validation_data))
    forecasts[:,0] = results
    '''
    for i in range(len(validation_data)):
        pred = temp_mod.predict(i+1)
        forecasts[i, 0] = pred
    '''
    temp_mod_score = MAE_error(forecasts)
    return temp_mod_score


#Search for optimal hyperparameters for the SARIMA-forecasting model
def grid_search_params(self, training_data, validation_data):
    candidate_model_params = {}
    for p in self.p_params:
        for q in self.q_params:
            for d in self.d_params:
                for P in self.P_params:
                    for Q in self.Q_params:
                        for D in self.D_params:
                            for m in self.m_params:
                                order_temp = (p, q, d)
                                seasonal_order_temp = (P, Q, D, m)
                                temp_mod = SARIMAX(training_data, order=order_temp, seasonal_order=seasonal_order_temp)
                                score = eval_model_for_grid_search(temp_mod, validation_data)
                                candidate_model_params[score] = (order_temp, seasonal_order_temp)
    top_score = min(candidate_model_params.keys())
    best_params = candidate_model_params[top_score]
    print(best_params)
    return best_params



class SARIMA_model:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters):
        self.model = None
        self.order = parameters["SARIMA_order"]
        self.seasonal_order = parameters["SARIMA_seasonal_order"]
        self.grid_search = parameters["SARIMA_grid_search"]
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
        self.validation_data = None
        return self


    def fit(self, features):        #, targets, batch_size, epochs, validation_split, shuffle, verbose):
        self.training_data = features[-(self.thershold + self.prediction_horizon): -self.prediction_horizon]
        self.validation_data = features[-self.prediction_horizon: ]
        if self.parameters_selected:
            self.model = SARIMAX(self.data, order=self.order, seasonal_order=self.seasonal_order)
            ret = self.model.fit()
        else:
            if self.grid_search:
                #must perform a grid search here...
                training_data = self.training_data
                validation_data = self.validation_data
                order, seasonal_order = grid_search_params(self, training_data, validation_data)
                self.order = order
                self.seasonal_order = seasonal_order
                self.model = SARIMAX(training_data, order=self.order, seasonal_order=self.seasonal_order)
                ret = self.model.fit()
                self.parameters_selected = True
            else:
                self.model = SARIMAX(training_data, order=self.order, seasonal_order=self.seasonal_order)
                ret = self.model.fit()
                self.parameters_selected = True
        return ret

'''
    def predict(self, test_data):
        predictions = np.zeros((len(test_data), self.prediction_horizon))
        for i in range(len(test_data)):
            
        return predictions

'''