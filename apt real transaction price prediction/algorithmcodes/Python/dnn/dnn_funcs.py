from dnn_libs import *

np.random.seed(1)  # for reproducibility

# test.csv load
def loadTestCSV(test_csv):
    data = pd.read_csv(test_csv)

    # dataframe value change
    data['front_door_structure'] = data['front_door_structure'].apply({'corridor': 0, 'stairway': 2, 'mixed': 1}.get)
    data['heat_fuel'] = data['heat_fuel'].apply({'gas': 1, 'cogeneration': 0}.get)
    data['heat_type'] = data['heat_type'].apply({'individual': 2, 'central': 0, 'district': 1}.get)

    # column dtype change
    data['city'] = data['city'].astype('category', copy=False)
    data['heat_fuel'] = data['heat_fuel'].astype('category', copy=False)
    data['heat_type'] = data['heat_type'].astype('category', copy=False)
    data['transaction_month'] = data['transaction_month'].astype('category', copy=False)
    data['year_of_completion'] = data['year_of_completion'].astype('category', copy=False)
    data['front_door_structure'] = data['front_door_structure'].astype('category', copy=False)
    data['address_by_law_first5'] = data['address_by_law_first5'].astype('category', copy=False)

    # unnecessary column drop
    data = data.drop(['Unnamed: 0', 'real_price'], axis=1)
    return data

# train.csv load
def loadTrainCSV(train_csv):
    data = pd.read_csv(train_csv)

    # column dtype change
    data['city'] = data['city'].astype('category', copy=False)
    data['heat_fuel'] = data['heat_fuel'].astype('category', copy=False)
    data['heat_type'] = data['heat_type'].astype('category', copy=False)
    data['transaction_month'] = data['transaction_month'].astype('category', copy=False)
    data['year_of_completion'] = data['year_of_completion'].astype('category', copy=False)
    data['front_door_structure'] = data['front_door_structure'].astype('category', copy=False)
    data['address_by_law_first5'] = data['address_by_law_first5'].astype('category', copy=False)

    # unnecessary column drop
    data = data.drop(['Unnamed: 0', 'key'], axis=1)
    return data

# split train.csv (to x_train, x_test, y_train, y_test)
def splitTraindata(train_csv, data_len, test_size):
    data = loadTrainCSV(train_csv)

    X = data.drop(['real_price'], axis=1)
    Y = data[['real_price']]

    X = X[:data_len]
    Y = Y[:data_len]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test

# main function
def main(train_csv, test_csv, data_len, test_size, shape, epochs, batch):
    # load Datas
    trainX_train, trainX_test, trainY_train, trainY_test = splitTraindata(train_csv, data_len, test_size)

    testX_test = loadTestCSV(test_csv)
    testX_test2 = testX_test.drop(['key'], axis=1)

    # create dnn model
    model = Sequential()

    model.add(Dense(128, input_shape=(shape,), kernel_initializer='normal'))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='linear'))

    # model.summary()
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX_train, trainY_train, epochs=epochs, batch_size=batch, verbose=1, validation_split=0.2)

    # r-squared, mse, rmse of train.csv
    prediction1 = model.predict(trainX_test)
    prediction1_r2 = r2_score(prediction1, trainY_test)
    prediction1_mse = mean_squared_error(prediction1, trainY_test)
    prediction1_rmse = np.sqrt(prediction1_mse)

    # prediction of test.csv
    prediction2 = model.predict(testX_test2)

    print('')
    print('r-squared : ', prediction1_r2, ', mse : ', prediction1_mse, ', rmse : ', prediction1_rmse) # r-squared, mse, rmse of train.csv
    print('')

    # value to list
    key_list = list(testX_test['key'])
    prediction2_list = list(prediction2)

    # list to df
    submission_data = {'key': key_list, 'transaction_real_price': prediction2_list}
    submission = pd.DataFrame.from_dict(submission_data, columns = ['key', 'transaction_real_price'])

    # df to csv
    submission.to_csv("submission.csv")
    return print('key : ', testX_test['key'], ', prediction : ', prediction2)  # key, prediction of test.csv