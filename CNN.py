from __future__ import annotations

import numpy as np


from keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Flatten,
    InputLayer,
    Conv1D,
    GlobalMaxPooling1D,
)
from keras.losses import KLDivergence
from keras.models import Sequential, load_model
from keras.models import Model
from keras import metrics
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.python.keras.backend import (
    binary_crossentropy,
    conv1d,
    categorical_crossentropy,
)
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.losses import BinaryCrossentropy

from tcn import TCN

# "learning_rate": 0.001,  # Learning rate the neural net
# activation functions: linear, sigmoid, tanh, or relu, need to be at least same lenght as hidden layers.
# "anet_optimizer": "Adam",  # Adam, SGD, Adagrad, RMSprop


class CNN:
    def __init__(self) -> None:
        return None

    def initialize(self, parameters, input_length):
        if parameters["starting_cnn"] != "":
            self.load_model(parameters["starting_cnn"])
            return self

        self.model = Sequential()

        self.model.add(
            InputLayer(input_shape=(parameters["training_length"] * 24, input_length))
        )  # need to add input shape here
        
        # adding the TCN layer here
        self.model.add(
            TCN(
                nb_filters=parameters["TCN_nb_filters"],
                kernel_size=parameters["TCN_kernel_size"],
                nb_stacks=parameters["TCN_nb_stacks"],
                dilations=parameters["TCN_dilations"],
                padding=parameters["TCN_padding"],
                dropout_rate=parameters["TCN_dropout_rate"],
                activation=parameters["TCN_activiation"],
            )
        )
        
        if len(parameters["hidden_layers"]) < len(parameters["activation_functions"]):
            print("NEED TO HAVE ACTIVATION FUNCTIONS EQUAL TO NUMBER OF HIDDEN LAYERS DUMBASS!")

        # adding dense layers
        for index, layer in enumerate(parameters["hidden_layers"]):
            self.model.add(
                Dense(layer, activation=parameters["activation_functions"][index])
            )

        # Will need to map to 24 * prediction_horizon floats in last layer
        self.model.add(Dense(24 * parameters["prediction_horizon"], activation=parameters["last_layer_activation"]))

        selected_optimizer = getattr(optimizers, parameters["optimizer"])(
            learning_rate=parameters["learning_rate"]
        )

        self.model.compile(
            loss=parameters["loss"],
            optimizer=selected_optimizer,
            metrics=parameters["metrics"],
        )

        self.model.summary()

        return self

    def predict(self, input):
        return self.model.predict(input)

    def fit(
        self, features, targets, batch_size, epochs, validation_split, shuffle, verbose
    ):

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

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)


# history = self.model.fit(x=features, y=targets, batch_size = how many to fit at a time, epochs = 5, validation_split = 0.1,shuffle = True) # how to train
# self.model( [input] ) -> output

