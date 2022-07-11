from .rf_funcs import *

csv = '' # 괄호 안에 가져올 csv file, ex) realestateDataForAnalysis6_noOutlier.csv
n_estimators = 1000 # 상수 값 수정!!!
max_depth = 5 # 상수 값 수정!!!
random_state = 0 # 상수 값 수정!!!
n_jobs = -1 # 상수 값 수정!!!

if __name__ == "__main__":
    rfMain(csv, n_estimators, max_depth, random_state, n_jobs)