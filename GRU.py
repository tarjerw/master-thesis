#Thinking about mmaking an GRU model for 
from keras.models import Sequential
from keras.layers import Dense, GRU, Flatten
import tensorflow.keras.optimizers as optimizers


class GRU_model:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters):

        self.model = Sequential()
        if len(parameters["GRU_output_units"]) <= 1:
            self.model.add(GRU(parameters["GRU_output_units"][0],
            activation=parameters["GRU_activation"],
            recurrent_activation=parameters["GRU_recurrent_activation"]))
        else:
            for i in range(len(parameters["GRU_output_units"]) - 1):
                self.model.add(GRU(parameters["GRU_output_units"][i],
                return_sequences=True, activation=parameters["GRU_activation"],
                recurrent_activation=parameters["GRU_recurrent_activation"],
                dropout=parameters["GRU_dropout"]))
            if parameters["GRU_return_sequences_last"]:
                self.model.add(GRU(parameters["GRU_output_units"][-1],
                return_sequences=True, activation=parameters["GRU_activation"],
                recurrent_activation=parameters["GRU_recurrent_activation"],
                dropout=parameters["GRU_dropout"]))
                self.model.add(Flatten()) #Returns hidden state for last layer as well, therefore, flatten is required
            else:
                self.model.add(GRU(parameters["GRU_output_units"][-1],
                return_sequences=False, activation=parameters["GRU_activation"],
                recurrent_activation=parameters["GRU_recurrent_activation"],
                dropout=parameters["GRU_dropout"]))

        for i in range(len(parameters["GRU_DNN_layers"])):
            self.model.add(Dense(parameters["GRU_DNN_layers"][i],
            activation=parameters["GRU_DNN_activation"]))
        
        self.model.add(Dense(parameters["prediction_horizon"]*24))

        selected_optimizer = getattr(optimizers, parameters["optimizer"])(
            learning_rate=parameters["learning_rate"]
        )

        self.model.compile(
            loss=parameters["loss"],
            optimizer=selected_optimizer,
            metrics=parameters["metrics"],
        )

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