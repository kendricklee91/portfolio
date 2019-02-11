from dnn_funcs import *

# csv files
train_csv = '/usingcsvfiles/train.csv'
test_csv = '/usingcsvfiles/test.csv'

# parameters to be used for train.csv
data_len = 1435827
test_size = 0.1

# parameters to be used for dnn model
shape = 63
epochs = 200
batch = 100


if __name__ == "__main__":
    start_time = time.time()
    main(train_csv, test_csv, data_len, test_size, shape, epochs, batch)
    end_time = time.time()

    print('========================================')
    print('Working time : {} sec'.format(end_time - start_time))