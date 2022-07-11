from .rf_libs import *

def getData(csv):
    data = pd.read_csv(csv)

    data['city'] = data['city'].astype("category", copy=False)
    data['heat_type'] = data['heat_type'].astype("category", copy=False)
    data['heat_fuel'] = data['heat_fuel'].astype("category", copy=False)
    data['transaction_month'] = data['transaction_month'].astype("category", copy=False)
    data['front_door_structure'] = data['front_door_structure'].astype('category', copy = False)
    data['address_by_law_first5'] = data['address_by_law_first5'].astype("category", copy=False)

    data = data.dropna().drop(['Unnamed: 0'], axis = 1)
    return data

def splitData(csv):
    data = getData(csv)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    return x_train, x_test, y_train, y_test

def rfMain(csv, n_estimators, max_depth, random_state, n_jobs):
    x_train, x_test, y_train, y_test = splitData(csv)

    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, n_jobs = n_jobs)
    model.fit(x_train.values, y_train.values)

    pred = model.predict(x_test)

    rf_r2 = r2_score(pred, y_test)
    rf_mse = mean_squared_error(pred, y_test)
    rf_rmse = np.sqrt(rf_mse)
    return print('r2 : ', rf_r2, ', mse : ', rf_mse, ', rmse : ', rf_rmse)