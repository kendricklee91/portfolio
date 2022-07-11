from .ridge_funcs import *

csv = '' # 괄호 안에 가져올 csv file, ex) realestateDataForAnalysis6_noOutlier.csv
alphas = 10 ** np.linspace(10, -2, 100) * 0.5

if __name__ == "__main__":
    ridgeMain(csv, alphas)