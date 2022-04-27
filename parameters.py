import tensorflow as tf

output_variable = ( 
    "Oslo"  # what are we forecasting, in this thesis "System Price"
)

selected_colums = [
    "Oslo",
    "Kr.sand",
    "Tr.heim",
    "Tromsø",
    "Bergen",
    "Month",
    "Weekday",
] 

selected_colums_regression = [
    "System Price",
    "Oslo",
    "Kr.sand",
    "Tr.heim",
    "Tromsø",
   # "Bergen",
   #"Month",
    #"Weekday",
   # "Holiday",
] 




# ONLY HERE CHANGES TO THE MODEL NEED TO BE DONE (and two vars in data_processing)
parameters = {
    # IF TO LOAD OLD CNN OR NOT!
    # Path for continuing training (e.g "/Users/tarje/Desktop/Fordypningoppgvae/git/models/6/11.01.2021/13.39.29, "" for no pre-load)
    "starting_cnn": "",
    # params for CNN:
    "epochs": 3,
    "batch_size": 256,  # batch size
    "validation_split": 0.1,
    "learning_rate": 0.00000002,  # Learning rate the neural net
    "hidden_layers": [32, 16,8],  # Hidden layers for CNN
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
    "TCN_nb_filters": 16,  # the more the better, but be wary of overfitting at some point
    "TCN_kernel_size": 16,  #  how many time steps considered (depend on how long back the data is dependent), important param, can't be longer than training_lenght
    "TCN_nb_stacks": 1,  # Not very useful unless your sequences are very long (like waveforms with hundreds of thousands of time steps)
    "TCN_dilations": (
        1,
        2,
        4,
        8,
        16,
    ),  # It controls how deep your TCN layer is. Usually, consider a list with multiple of two
    "TCN_padding": "causal",  # causal prevents information leakage (keep as is)
    "TCN_dropout_rate": 0.01,  # can be used to reduce overfitting (0 or lower than 0.05)
    "TCN_activiation": "relu",  # leave to default (relu)
    
    # from data processing (must also be changed there)
  
    "training_length": 7,
    "output_variable": "Tromsø",
    "prediction_horizon": 7,
    "selected_colums": selected_colums,
    "output_variable": output_variable,
    "base_model": "regression", # "naive", "regression"
    "regression_poly": 2, # what factor of polynomials in regression (1 = linear)
    "enhanced_naive": True, # enhanced naive vs. naive 
    "standardize_data": True, # method of preprocessing 
    "min_max_normalize_data": False, # method of preprocessing 
    "test_split": '2020-01-01-0',# first hour in test set

    "TCN_factor": 1.00,  # factor multiplied with TCN effect on price, set to 0.0 if want to test naive/ enhanced naive, else (0.75)
    
    "selected_colums_regression": selected_colums_regression,



    # setting random seed
    "tenserflow_random_seed": 69,
    "numpy_random_seed": 420,

    #

}

"""
The test case study is designed by using data from 1 Jan 2014 to 31 Dec 2019 as training data and 1 Jan 2020 to
31 Dec 2020 as testing data. 

VARIABLES: 
# every class not spesified is a float. Everything except Date is float or int. Converted selected columns to floats

TIME:
    "Date", (2014-01-01, str)
    "Week", (1, int) (converted to one-hot encoding, but very long, leads to 53 input vars)
    "Month", (1, int) (converted to one-hot encoding)
    "Season", (1, int, 1-4) (rather use month as it incorporates it)
    "Weekday", (3, int, 1-7) (converted to one-hot encoding)
    "Weekend", (0, int, 0-1) (rather use weekday as it incorporates it)
    "Holiday", (1, int, 0-1)
    "Hour", (0-23, int) (converted to one-hot encoding)
PRICE DATA:
    "System Price", (28.25, EUR / MWh)
    "Oil", (81.12, eur per barrel crude oil)
    "Coal", (59.71)
    "Gas",(28.11)
    "Low Carbon", (100.32)
    "APX", (41.48)
    "OMEL", (12.86)
    "EEX", (17.38)
Regional Prices: (THESE ARE NEW!)
    "SE1", (EUR / MWh)
    "SE2", (EUR / MWh)
    "SE3", (EUR / MWh)
    "SE4", (EUR / MWh)
    "DK1", (EUR / MWh)
    "DK2", (EUR / MWh)
    "FI", (EUR / MWh)
    "Oslo", (EUR / MWh)
    "Kr.sand", (EUR / MWh)
    "Bergen", (EUR / MWh)
    "Tr.heim", (EUR / MWh)
    "Tromsø", (EUR / MWh)
    "Molde", (EUR / MWh) # IGNORE MOLDE, same as Tr.heim
MARKET DATA
    "Total Vol", (884300.0)
    "NO Buy Vol", (340886, int)
    "NO Sell Vol", (299489, int)
    "SE Buy Vol", (355291, int)
    "SE Sell Vol", (408669, int)
    "DK Buy Vol", (58534.4)
    "DK Sell Vol", (101167.0)
    "FI Buy Vol", (134389, int)
    "FI Sell Vol", (98827.4)
    "Nordic Buy Vol", (889099.0)
    "Nordic Sell Vol", (908152.0)
    "Baltic Buy Vol", (42944.1)
    "Baltic Sell Vol", (23891.8)
    "Supply", (1076510.0)
    "Demand", (1117970.0)
PRODUCTION DATA:
    "NO Hydro", (54821.7)
    "SE Hydro", (21822.1)
    "FI Hydro", (3736.57)
    "Total Hydro", (80380.4)
    "NO Hydro Dev", (-1803.51)
    "SE Hydro Dev", (1589.92)
    "FI Hydro Dev", (209.37)
    "Total Hydro Dev", (-4.24)
    "Wind DK", (46140, int)
    "Wind SE", (93454.6)
    "Wind Prod", (139595.0)
WEATHER DATA (only Norway):
    "T Nor", (4.4, celsius)
    "Temp Hamar", (temp in celsius, float) Oslo
    "Temp Krsand", (temp in celsius, float) Kr.Sand
    "Temp Namsos", (temp in celsius, float) Tr.heim
    "Temp Troms", (temp in celsius, float) Tromsø
    "Temp Bergen", (temp in celsius, float) Bergen
    "Temp Norway", (temp in celsius, float)
    "Prec Norway", (101.2)
    "Prec Norway 7", (981.4)
    "Snow Norway", (45.0983)
Marius & Jake Estimate:
    "Curve Demand", (884300.0) don't think we are using this
"""