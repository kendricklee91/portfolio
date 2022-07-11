from .dt_funcs import *

csv = '' # 괄호 안에 가져올 csv file, ex) realestateDataForAnalysis6_noOutlier.csv
max_depth = 5
random_state=0

if __name__ == "__main__":
    dtMain(csv, max_depth, random_state)