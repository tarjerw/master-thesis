#Maalet her er aa returnere trenings- og test-data med en tilhoerende "path" for n_dager x 24 t
#Normalization and other adoptions to the data can be made in another script

from cgi import print_exception
import numpy as np
import pandas as pd
from coeff_tree import Node
import matplotlib.pyplot as plt
from tqdm import tqdm


#data  = pd.read_csv('data_erling/hourly_data_areas.csv')
price_columns_main = ['System Price', 'SE1', 'SE2', 'SE3', 'SE4', 'FI', 'DK1', 'DK2', 'Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim', 'Tromsø']
coeff_columns_main = ['Month', 'Weekday', 'Weekend', 'Hour', 'Holiday']

def read_data(columns, path='hourly_data_areas.csv'):
    data = pd.read_csv(path)
    data = data[columns]
    return data

def get_price_columns(cols):
    return [x for x in cols if x in price_columns_main]

def get_coeff_cols(cols):
    return [x for x in cols if x in coeff_columns_main]

def make_train_test_split(data, date_string, hr=0):
    index = data.index[(data['Date']==date_string) & (data['Hour']==hr)].to_list()[0]
    training_data = data.iloc[:index]
    test_data = data.iloc[index:]
    return training_data, test_data

#Calculates coefficients for a single price columbn, wrt. the different price columns included
def calc_coeff(data, coeff_node, price_node):
    avg = data[price_node.value].mean()
    coeffs = {}
    coeff_set = set(data[coeff_node.value])
    for coeff in coeff_set:
        coeff_avg = data[price_node.value].loc[data[coeff_node.value] == coeff].mean()
        coeffs[coeff] = coeff_avg/avg
    return coeffs

#Makes and calculates the coefficient tree - 4 levels - ROOT - PRICE AREA - TIME DEPENDEND COEFFICIENT - ACTUAL COEFFICIENT
def make_coeff_tree(data, price_columns, coeff_columns):
    price_columns = [x for x in price_columns_main if x in price_columns]
    coeff_columns = [x for x in coeff_columns_main if x in coeff_columns]
    root = Node(root=True)
    area_node_list = []
    for area in price_columns:
        area_node = Node(value=area, parent=root)
        area_node_list.append(area_node)
        area_node.set_coeff_value = data[area].mean()
        area_node.set_parent(root)

    #Building the tree below the root node
    for area_node in area_node_list:
        for coeff in coeff_columns:
            coeff_node = Node(value=coeff, parent=area_node)
            coeff_node.set_parent(area_node)
            coeffs_for_area = calc_coeff(data, coeff_node, area_node)
            for coefficient_value_key in coeffs_for_area.keys():    #Coeff_value_key is eg. hr=2 or weekday=4
                coeff_value_node = Node(value=coefficient_value_key, parent=coeff_node)
                coeff_value_node.set_parent(coeff_node)
                coeff_value_node.set_coeff_value(coeffs_for_area[coefficient_value_key])
    return root


def get_coeff_from_tree(root, area, coeff, value):
    search_list = [area, coeff, value]
    return root.get_coefficient(search_list)


#Takes in ONE price area, and adjusts this price
def adjust_price_to_deviation(data, price_col, coeff_cols, tree_root):
    avg_price = data[price_col].mean()
    price_new = data[price_col].to_numpy()
    for i in range(len(price_new)):
        expected_price_this_point_in_time = avg_price
        for j in range(len(coeff_cols)):
            expected_price_this_point_in_time = expected_price_this_point_in_time * get_coeff_from_tree(tree_root, price_col, coeff_cols[j], data[coeff_cols[j]].iloc[i])
        price_new[i] -= expected_price_this_point_in_time
    #Now, this function returns a numpy array, should be made into a pandas series or somethingg
    #print(data[price_col])
    #print(price_new)
    return price_new, avg_price


#Now, the prices will represent the deviation from the enhanced naive expectation of this point in time, based on the values which we have available. A dictionary containing the average prices for the different areas will also be returned
def transform_prices(data, price_cols, coeff_cols, tree_root):
    price_avg = {}
    for price in price_cols:
        new_price, avg = adjust_price_to_deviation(data, price, coeff_cols, tree_root)
        price_avg[price] = avg
        data[price] = new_price
    return data, price_avg

def plot_data(data, cols):
    plt.style.use('ggplot')
    for col in cols:
        plt.plot(data[col])
   #plt.show()



#returns data on the shape [n_instances, n_timesteps_back, n_features], [n_instances, target_vector]
def make_data_to_sheets(data, input_features, output_variables, input_length, output_length):
    X_data_sheets = np.zeros((data.shape[0] - output_length, input_length, len(input_features)))
    y_data_sheets = np.zeros((data.shape[0] - output_length, output_length))
    print('making data sheets...')
    for i in tqdm(range(len(data) - output_length - input_length)):
        sheet = data[input_features].iloc[i : i + input_length]
        target = data[output_variables].iloc[i + input_length : i + input_length + output_length]
        #print(sheet)
        #print(target)
        X_data_sheets[i, :, :] = sheet
        y_data_sheets[i, :] = target.to_numpy().reshape((output_length, ))
    return X_data_sheets, y_data_sheets




'''
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
        
        1. Omstille dataen. Da må vi ha både et transformasjonstre og et gjennomsnitt for å kunne gjøre om den transformerte dataen til opprinnelig data
            Legge inn et eget tre som har mapping fra Område -> avg evt kan dette faktisk bare være en dictionary, siden det kun er ett nivå
        2. Lage sheets av dataen - det er dette som gjør at en eks LSTM kan bruke det. [samples, timesteps, features]


        train_data, test_data = adjust_data_with_tree(tree_root, train_data, test_data)
        #Should consider making the time series to deviations from average prices 
        #X_train, y_train, X_test, y_test = make_sheets(input_length, pred_length, target_col, train_data, test_data)
        #return data, X_train, y_train, X_test, y_test #Some kind of baseline/enhanced naive etc. must be return in order for the model to make predictions
        #make the sheets for each prediction. The sheet is (n_features, n_timesteps behind)
        #Return (data, X_train, y_train, X_test, y_test)

'''

if __name__=='__main__':
    data = read_data(['Hour', 'System Price', 'SE1', 'SE2', 'SE3', 'SE4']) #'Unnamed: 0' and 'Date' has been removed, among others
    price_columns_as_coefficients = ['System Price', 'SE1', 'SE2']
    coeff_columns_as_coefficients = ['Hour']
    root_node = make_coeff_tree(data, price_columns_as_coefficients, coeff_columns_as_coefficients)
    #root_node.print_subtree()
    #plot_data(data[:200], ['SE2'])
    data, average_area_prices = transform_prices(data, price_columns_as_coefficients, coeff_columns_as_coefficients, root_node)
    X_data_sheet, y_data_sheet = make_data_to_sheets(data[:15000], data.columns, ['SE2'], 168, 24) #MODIFICATION IS REQUIRED, IS INCONSISTENT...
    print(y_data_sheet[0].shape)
    #plot_data(data[:200], ['SE2'])
    #plt.show()
    #print(data[['System Price', 'SE1', 'SE2']].describe())


