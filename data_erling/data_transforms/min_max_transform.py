from sklearn.preprocessing import MinMaxScaler

class Min_max_scaler:

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()
        pass

    def fit_scaler(self, data) -> None:
        self.scaler.fit(data)

    def transform_data_with_minmax(self, data):
        return self.scaler.transform(data)