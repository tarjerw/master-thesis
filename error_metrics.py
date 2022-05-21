import numpy as np

# Will contain the following error metrics: RMSE, MAPE and MAE
# Input for all will be a numpy array containing tuples with [forecast, target]


def MAE_error(data):  # good

    return np.average(list(np.abs(x[0] - x[1]) for x in data))


def MAPE_error(data):  # good

    output_sum = 0
    observations = 0

    for x in data:
        if x[1] >= 1:
            devi = np.abs(x[1] - x[0]) / x[1]
            output_sum += devi
            observations += 1

    return 100 * (output_sum / observations)


def RMSE_error(data):

    output_sum = 0
    for x in data:
        output_sum += (x[0] - x[1]) ** 2
    output_sum = output_sum / len(data)
    return output_sum ** 0.5


def SMAPE_error(data):
    output_sum = 0
    observations = 0
    for x in data:
        if x[1] >= 1:
            top = np.abs(x[1] - x[0])
            bottom = (np.abs(x[1]) + np.abs(x[0])) / 2
            output_sum += top / bottom
            observations += 1
    return 100 * (output_sum / observations)


def get_metrics(test_list):
    metrics = {
        "MAE": MAE_error(test_list),
        "MAPE": MAPE_error(test_list), 
        "RMSE": RMSE_error(test_list),
        "SMAPE": SMAPE_error(test_list),
    }
    return metrics