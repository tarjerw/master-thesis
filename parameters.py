import tensorflow as tf
from data_processing import (
    input_length,
    training_length,
    output_variable,
    prediction_horizon
)


# ONLY HERE CHANGES TO THE MODEL NEED TO BE DONE (and two vars in data_processing)
parameters = {
    # IF TO LOAD OLD CNN OR NOT!
    # Path for continuing training (e.g "/Users/tarje/Desktop/Fordypningoppgvae/git/models/6/11.01.2021/13.39.29, "" for no pre-load)
    "starting_cnn": "",
    # params for CNN:
    "epochs": 5,
    "batch_size": 88,  # batch size
    "validation_split": 0.1,
    "learning_rate": 0.00000000000000000000000000000002,  # Learning rate the neural net
    "hidden_layers": [64, 32,32],  # Hidden layers for CNN
    "activation_functions": [
        "relu",
        "relu",
        "relu"
    ],  # activation functions for hidden layers, (can't be shorter than "hidden_layers")
    # linear, sigmoid, tanh, or relu, need to be at least same lenght as hidden layers.
    "last_layer_activation": "linear",
    "optimizer": "Adam",  # Adam, SGD, Adagrad, RMSprop
    "loss": "mean_squared_error",  # loss function in CNN, "mean_absolute_error", "mean_squared_error"
    "verbose": 1,  # 0,1 or 2, affects data feedback while training (no impact on result)
    "metrics": [
        "MAE",
        "MAPE",
        tf.keras.metrics.RootMeanSquaredError(),
        "binary_crossentropy"
    ],  # (no impact on result)
    # TCN parameters:
    "TCN_nb_filters": 32,  # the more the better, but be wary of overfitting at some point
    "TCN_kernel_size": 32,  #  how many time steps considered (depend on how long back the data is dependent), important param, can't be longer than training_lenght
    "TCN_nb_stacks": 1,  # Not very useful unless your sequences are very long (like waveforms with hundreds of thousands of time steps)
    "TCN_dilations": (
        1,
        2,
        4,
        8,
        16,
        32,
    ),  # It controls how deep your TCN layer is. Usually, consider a list with multiple of two
    "TCN_padding": "causal",  # causal prevents information leakage (keep as is)
    "TCN_dropout_rate": 0.00,  # can be used to reduce overfitting (0 or lower than 0.05)
    "TCN_activiation": "relu",  # leave to default (relu)
    # from data processing (must also be changed there)
    "input_length": input_length,
    "training_lenght": training_length,
    "output_variable": output_variable,
    "prediction_horizon": prediction_horizon,
    "TCN_factor": 0.00,  # factor multiplied with TCN effect on price, set to 0.0 if want to test naive/ enhanced naive, else (0.75)
    # setting random seed
    "tenserflow_random_seed": 69,
    "numpy_random_seed": 420,
}