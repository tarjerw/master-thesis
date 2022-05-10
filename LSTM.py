#Thinking about mmaking an LSTM model for 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten


class LSTM_model:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters):

        self.model = Sequential()
        if len(parameters["LSTM_output_units"]) <= 1:
            self.model.add(LSTM(parameters["LSTM_output_units"][0],
            activation=parameters["LSTM_activation"],
            recurrent_activation=parameters["LSTM_recurrent_activation"]))
        else:
            for i in range(len(parameters["LSTM_output_units"])):
                self.model.add(LSTM(parameters["LSTM_output_units"][i],
                return_sequences=True, activation=parameters["LSTM_activation"],
                recurrent_activation=parameters["LSTM_recurrent_activation"],
                dropout=parameters["LSTM_dropout"]))

        self.model.add(Flatten())#should discuss with BVey

        for i in range(len(parameters["LSTM_DNN_layers"])):
            self.model.add(Dense(parameters["LSTM_DNN_layers"][i],
            activation=parameters["LSTM_DNN_activation"]))
        
        self.model.add(Dense(parameters["prediction_horizon"]*24))

        self.model.compile(loss=parameters["loss"],
        metrics=parameters["metrics"],
        optimizer=parameters["optimizer"])

        #self.model.summary()

        return self



    def fit(
        self, features, targets, batch_size, epochs, validation_split, shuffle, verbose
    ):
        print(features.shape)
        print(targets.shape)

        ret = self.model.fit(
            x=features,
            y=targets,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            shuffle=shuffle,
            verbose=verbose,
        )

        # loss = ret.history.get("loss", [None])
        # mae = ret.history.get("MAE", [None])

        return ret

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, path):
        self.model.save(path)

'''
    def load_model(self, path):
        self.model = loa'''