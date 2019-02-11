from .lasso_libs import *

def getData(csv):
    data = pd.read_csv(csv)

    data['city'] = data['city'].astype("category", copy=False)
    data['heat_type'] = data['heat_type'].astype("category", copy=False)
    data['heat_fuel'] = data['heat_fuel'].astype("category", copy=False)
    data['transaction_month'] = data['transaction_month'].astype("category", copy=False)
    data['year_of_completion'] = data['year_of_completion'].astype("category", copy=False)
    data['address_by_law_first5'] = data['address_by_law_first5'].astype("category", copy=False)

    data = data.dropna().drop(['Unnamed: 0'], axis = 1)
    return data

def splitData(csv):
    data = getData(csv)

    X = data.drop(['real_price'], axis=1)
    Y = data[['real_price']]

    x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    return x_train, x_test, y_train, y_test

def lassoMain(csv, cv, max_iter):
    x_train, x_test, y_train, y_test = splitData(csv)

    lassocv = LassoCV(alphas = None, cv = cv, max_iter = max_iter, normalize = True, verbose = True)
    lassocv.fit(x_train, y_train)

    lasso = Lasso(max_iter = max_iter, normalize = True)
    lasso.set_params(alpha = lassocv.alpha_)
    lasso.fit(x_train, y_train)

    pred = lasso.predict(x_test)

    mse = mean_squared_error(pred, y_test)
    rmse = np.sqrt(mse)
    return print('mse : ', mse, ', rmse : ', rmse)