from .lasso_funcs import *

csv = '' # 괄호 안에 가져올 csv file, ex) realestateDataForAnalysis6_noOutlier.csv
cv = 10 # 상수 값 수정!!!
max_iter = 10000 # 상수 값 수정!!!

if __name__ == "__main__":
    lassoMain(csv, cv, max_iter)
