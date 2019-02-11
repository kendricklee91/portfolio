from .svr_libs import *

def getData(csv):
    data = pd.read_csv(csv)

    data['heat_fuel'] = data['heat_fuel'].apply({'gas': 0, 'cogeneration': 1}.get)
    data['heat_type'] = data['heat_type'].apply({'individual': 0, 'central': 1, ' district': 2}.get)
    data['front_door_structure'] = data['front_door_structure'].apply({'corridor': 0, 'stairway': 1, 'mixed': 2}.get)


    data['city'] = data['city'].astype('category', copy = False)
    data['heat_fuel'] = data['heat_fuel'].astype('category', copy = False)
    data['heat_type'] = data['heat_type'].astype('category', copy = False)
    data['transaction_month'] = data['transaction_month'].astype('category', copy = False)
    data['year_of_completion'] = data['year_of_completion'].astype('category', copy = False)
    data['front_door_structure'] = data['front_door_structure'].astype('category', copy = False)
    data['address_by_law_first5'] = data['address_by_law_first5'].astype('category', copy = False)

    data = data.drop(['Unnamed: 0'], axis = 1)
    return data

def splitData(csv):
    data = getData(csv)

    X = data.drop(['real_price'], axis = 1)
    Y = data[['real_price']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    return x_train, x_test, y_train, y_test

def svrMain(csv, epsilon):
    x_train, x_test, y_train, y_test = splitData(csv)

    model = LinearSVR(epsilon = epsilon)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    svr_r2 = r2_score(pred, y_test)
    svr_mse = mean_squared_error(pred, y_test)
    svr_rmse = np.sqrt(svr_mse)
    return print('r2 : ', svr_r2, ', mse : ', svr_mse, ', rmse : ', svr_rmse)