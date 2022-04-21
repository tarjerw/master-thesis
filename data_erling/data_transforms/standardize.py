from sklearn.preprocessing import StandardScaler

class Standardizer:

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        pass

    def fit_scaler(self, data) -> None:
        self.scaler.fit(data)

    def transform_data_with_standardizer(self, data):
        return self.scaler.transform(data)
    