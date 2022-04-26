

week_day_coefficients = {
    1: 1.028603,
    2: 1.034587,
    3: 1.0301834,
    4: 1.033991,
    5: 1.014928,
    6: 0.941950,
    7: 0.915758,
}
holiday_coefficient = 1 / 0.89

#future_day_weekday = daily_data.iloc[day_index]["Weekday"]
#current_weekday = future_day_weekday

def make_forecasts(start_date, number_of_days):
    forecast_start = date_hour_list.index(start_date)
    
    
    predictions = []
    for i in range(number_of_days):
        predictions.extend(mlr_models[i].predict(forecast_basis_day))
    return predictions