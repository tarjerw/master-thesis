from __future__ import annotations

import tensorflow as tf

import numpy as np
from numpy.random import seed

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime
import json 
import pickle

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

from CNN import CNN

from parameters import parameters  # here changes to be model are done!!


from error_metrics import MAE_error, MAPE_error, RMSE_error, SMAPE_error

from data_processing import (
    selected_colums,
    input_length,
    training_length,
    prediction_horizon,
    date_hour_list,
    test_x,
    test_y,
    training_x,
    training_y
)

from tqdm import trange

# setting random seed for numpy and tensorflow XD
seed(parameters["numpy_random_seed"])
tf.random.set_seed(parameters["tenserflow_random_seed"])

# creating TCN/ CNN model based on parameters
model = CNN().initialize(parameters)

print(training_x.shape)

print(training_y.shape)

history = model.fit(
    features=training_x,
    targets=training_y,
    batch_size=parameters["batch_size"],
    epochs=parameters["epochs"],
    validation_split=parameters["validation_split"],
    shuffle=False,
    verbose=parameters["verbose"],
)


