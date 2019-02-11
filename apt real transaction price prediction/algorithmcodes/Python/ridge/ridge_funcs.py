from .ridge_libs import *

def getData(csv):
    data = pd.read_csv(csv)

    data = pd.get_dummies(data[['front_door_structure_category', 'heat_type_category', 'heat_fuel_category']])

    data['city_category'] = data['city'].astype("category")
    data['heat_type_category'] = data['heat_type'].astype("category")
    data['heat_fuel_category'] = data['heat_fuel'].astype("category")
    data['front_door_structure_category'] = data['front_door_structure'].astype("category", copy=False)
    data['address_by_law_first5_category'] = data['address_by_law_first5'].astype("category")
    data['transaction_month_category'] = data['transaction_month'].astype("category")

    data = data.dropna().drop(['heat_type', 'heat_fuel', 'front_door_structure','address_by_law_first5',
                               'transaction_month', 'Unnamed: 0', 'city', 'heat_type_category', 'heat_fuel_category',
                               'front_door_structure_category'], axis = 1)
    return data

def splitData(csv):
    data = getData(csv)

    X = data.drop(['real_price'], axis = 1)
    Y = data[['real_price']]

    x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    return x_train, x_test, y_train, y_test


def ridgeMain(csv, alphas):
    x_train, x_test, y_train, y_test = splitData(csv)

    ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
    ridgecv.fit(x_train, y_train)
    ridgecv.alpha_

    ridge = Ridge(alpha = ridgecv.alpha_, normalize = True)
    ridge.fit(x_train, y_train)

    pred = ridge.predict(x_test)

    mse = mean_squared_error(pred, y_test)
    rmse = np.sqrt(mse)

    return print('alpha : ', ridgecv.alpha_, ', predict : ', pred, ', mse : ', mse, ', rmse : ', rmse)

