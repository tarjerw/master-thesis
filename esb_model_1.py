#Thinking about mmaking an LSTM model for 
from keras.models import Sequential
from keras.layers import Dense, LSTM


class LSTM_model:
    def __init__(self, output_data_shape, LSTM_layers=[16], dense_layers=[32, 32, 16],loss_metric='mean_squared_error', accuracy_metrics=['accuracy'], optimizer='SGD') -> None:
        self.model = Sequential()
        self.model.add(LSTM(LSTM_layers[0]))
        '''
        #Layers - enabling stacked LSTM-models
        if len(LSTM_layers) > 1:
            for i in range()
        '''
        for i in range(dense_layers):
            self.model.add(Dense(dense_layers[i]))
        
        self.model.add(Dense(output_data_shape))

        self.model.compile(loss=loss_metric, metrics=accuracy_metrics, optimizer=optimizer)

    def fit_model(self, X_train, y_train, epochs=100):
        self.model.fit(X_train, y_train, epochs=epochs)