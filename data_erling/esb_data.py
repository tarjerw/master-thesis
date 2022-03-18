#Maalet her er aa returnere trenings- og test-data med en tilhoerende "path" for n_dager x 24 t
#Normalization and other adoptions to the data can be made in another script

import pandas as pd
from coeff_tree import Node


#data  = pd.read_csv('data_erling/hourly_data_areas.csv')
price_columns_main = ['System Price', 'SE1', 'SE2', 'SE3', 'SE4', 'FI', 'DK1', 'DK2', 'Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim', 'Tromsø']
coeff_columns_main = ['Month', 'Weekday', 'Weekend', 'Hour', 'Holiday']

def read_data(columns, path='hourly_data_areas.csv'):
    data = pd.read_csv(path)
    data = data[columns]
    return data

def make_train_test_split(data, date_string, hr=0):
    index = data.index[(data['Date']==date_string) & (data['Hour']==hr)].to_list()[0]
    training_data = data.iloc[:index]
    test_data = data.iloc[index:]
    return training_data, test_data

#Calculates coefficients for a single price columbn, wrt. the different price columns included
def calc_coeff(data, coeff_col, price_col):
    avg = data[price_col.value].mean()
    coeffs = {}
    coeff_set = set(data[coeff_col])
    for coeff in coeff_set:
        coeff_avg = data[price_col.value].loc[data[coeff_col] == coeff].mean()
        coeffs[coeff] = coeff_avg/avg
    return coeffs

#Makes and calculates the coefficient tree - 4 levels - ROOT - PRICE AREA - TIME DEPENDEND COEFFICIENT - ACTUAL COEFFICIENT
def make_coeff_tree(data, price_columns, coeff_columns):
    price_columns = [x for x in price_columns_main if x in price_columns]
    coeff_columns = [x for x in coeff_columns_main if x in coeff_columns]
    root = Node(root=True)
    node_temp_list = []
    for area in price_columns:
        node_temp = Node(value=area, parent=root)
        node_temp_list.append(node_temp)
        node_temp.set_coeff_value = data[area].mean()
        node_temp.set_parent(root)
    for area in node_temp_list:
        for coeff in coeff_columns:
            node_temp = Node(value=coeff, parent=root)
            node_temp.set_parent(area)
            coeffs_for_area = calc_coeff(data, coeff, area)
            for coefficient_value_key in coeffs_for_area.keys():    #Coeff_value_key is eg. hr=2 or weekday=4
                node_temp_2 = Node(value=coefficient_value_key, parent=node_temp)
                node_temp_2.set_parent(node_temp)
                node_temp_2.set_coeff_value(coeffs_for_area[coefficient_value_key])
    return root


'''
def transform_prices(data, price_col, coeff_col, tree):
    numpy_price = 


#Returns the transformed training and test data, based on findings in the training data
def adjust_data_with_tree(tree_root, train_data, test_data):
    


#def make_sheets(input_length, pred_length, target_col training_data, test_data):




def get_data(path, date_split, hr_split, input_length, pred_length, columns, target_col):
    if target_col not in columns:
        print('Cannot predic column outside data_columns')
    else:
        data = read_data(path)
        train_data, test_data = make_train_test_split(data, date_split, hr=hr_split)
        price_columns = [x for x in price_columns_main if x in price_columns]
        coeff_columns = [x for x in coeff_columns_main if x in coeff_columns]
        tree_root = make_coeff_tree(train_data, price_columns, coeff_columns)
        train_data, test_data = adjust_data_with_tree(tree_root, train_data, test_data)
        #Should consider making the time series to deviations from average prices 
        #X_train, y_train, X_test, y_test = make_sheets(input_length, pred_length, target_col, train_data, test_data)
        #return data, X_train, y_train, X_test, y_test #Some kind of baseline/enhanced naive etc. must be return in order for the model to make predictions
        #make the sheets for each prediction. The sheet is (n_features, n_timesteps behind)
        #Return (data, X_train, y_train, X_test, y_test)

'''

if __name__=='__main__':
    data = read_data(['Unnamed: 0', 'Date', 'Hour', 'System Price', 'Total Vol', 'NO Buy Vol',
       'NO Sell Vol', 'SE Buy Vol', 'SE Sell Vol', 'DK Buy Vol', 'DK Sell Vol',
       'FI Buy Vol', 'FI Sell Vol', 'Nordic Buy Vol', 'Nordic Sell Vol',
       'Baltic Buy Vol', 'Baltic Sell Vol', 'T Hamar', 'T Krsand', 'T Namsos',
       'T Troms', 'T Bergen', 'T Nor', 'NO Hydro', 'SE Hydro', 'FI Hydro',
       'Total Hydro', 'NO Hydro Dev', 'SE Hydro Dev', 'FI Hydro Dev',
       'Total Hydro Dev', 'Week', 'Month', 'Season', 'Weekday', 'Weekend',
       'Wind DK', 'Curve Demand', 'Holiday', 'SE1', 'SE2', 'SE3', 'SE4', 'FI',
       'DK1', 'DK2', 'Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim',
       'Tromsø'])
    root_node = make_coeff_tree(data, ['System Price', 'SE1', 'SE2'], ['Hour', 'Weekday', 'Holiday'])
    root_node.print_subtree()
