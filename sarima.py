from cgi import test
from socketserver import ThreadingUnixDatagramServer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

class SARIMA:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters):
        self.model = None
        self.order = parameters["SARIMA_order"]
        self.seasonal_order = parameters["SARIMA_seasonal_order"]
        self.prediction_horizon = parameters["prediction_horizon"] * 24
        self.target_variable = parameters["output_variable"]
        return self
    
    def fit(self, target):
        self.model = SARIMAX(target, order=self.order, seasonal_order=self.seasonal_order)
        ret = self.model.fit()
        return ret

    def predict(self, test_data):
        predictions = np.zeros((len(test_data), self.prediction_horizon))
        for i in range(len(test_data)):


