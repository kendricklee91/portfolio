from .linear_libs import *
def getData(csv):
    data = pd.read_csv(csv)


    data['heat_fuel'] = data['heat_fuel'].apply({'gas': 0, 'cogeneration': 1}.get)
    data['heat_type'] = data['heat_type'].apply({'individual': 0, 'central': 1, 'district': 2}.get)
    data['front_door_structure'] = data['front_door_structure'].apply({'corridor': 0, 'stairway': 1, 'mixed': 2}.get)

    data['city'] = data['city'].astype('category', copy=False)
    data['heat_fuel'] = data['heat_fuel'].astype('category', copy=False)
    data['heat_type'] = data['heat_type'].astype('category', copy=False)
    data['address_by_law'] = data['address_by_law'].astype('category', copy=False)
    data['transaction_month'] = data['transaction_month'].astype('category', copy=False)
    data['year_of_completion'] = data['year_of_completion'].astype('category', copy=False)
    data['front_door_structure'] = data['front_door_structure'].astype('category', copy=False)

    data = pd.DataFrame(data[['city', 'year_of_completion', 'exclusive_use_area', 'floor', 'address_by_law', 'total_parking_capacity_in_site',
                              'total_household_count_in_sites', 'apartment_building_count_in_sites', 'tallest_building_in_sites',
                              'lowest_building_in_sites', 'heat_type', 'heat_fuel', 'room_id', 'supply_area', 'total_household_count_of_area_type',
                              'bathroom_count', 'transaction_real_price', 'room_count', 'leading_index', 'coincident_index', 'lagging_index', 'transaction_month']])
    return data

def splitData(csv):
    data = getData(csv)

    X = data.drop(['transaction_real_price'], axis = 1)
    Y = data[['transaction_real_price']]

    x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    return x_train, x_test, y_train, y_test


def lmMain(csv):
    x_train, x_test, y_train, y_test = splitData(csv)

    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)

    pred = lm.predict(x_test)

    mse = mean_squared_error(pred, y_test)
    rmse = np.sqrt(mse)

    return print('mse : ', mse, ', rmse : ', rmse)

