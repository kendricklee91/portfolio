from .svr_funcs import *

csv = '' # 괄호 안에 가져올 csv file, ex) realestateDataForAnalysis6_noOutlier.csv
epsilon = 1.5

if __name__ == "__main__":
    svrMain(csv, epsilon)