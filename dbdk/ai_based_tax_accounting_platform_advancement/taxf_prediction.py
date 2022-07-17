from taxf_preprocessing import preprocessing
from taxf_data_loader import get_daily_data

import taxf_config as config
import taxf_utils as utils
import pandas as pd
import numpy as np
import argparse
import joblib
import copy

def predict(date = None):
    # 일자별 데이터 예측
    
    df = get_daily_data(date = date)
    outcome = ['yessin2', 'bill1out', 'bill2out', 'bill4out', 'card4out', 'card5out', 'cash3out', 'cash4out', 'cash5out', 'home1out', 'home2out', 'home3out', 'leaseout']

    df_assign = df.query(f"CD_TRAN in {outcome}").reset_index(drop = True)
    df_assgin.loc[:, 'pred_class_account'] = '401'
    df_assign.loc[:, 'probability_account'] = 1
    
    df_predict = df.query(f"CD_TRAN not in {outcome}").reset_index(drop = True)

    test_data = preprocessing(df_predict, inference = True)
    
    model_account = joblib.load(f"{config.MODEL_PATH}/model_cd_account_{config.WORD_FREQ}.pkl")
    model_dedu = joblib.load(f"{config.MODEL_PATH}/model_cd_dedu_{config.WORD_FREQ}.pkl")

    thresh_account = joblib.load(f"{config.MODEL_PATH}/thresh_per_cls_cd_account_{config.WORD_FREQ}.pkl")
    thresh_dedu = joblib.load(f"{config.MODEL_PATH}/thresh_per_cls_cd_dedu_{config.WORD_FREQ}.pkl")

    df_with_result = copy.deepcopy(df_predict)
    df_with_result.loc[:, 'pred_class_account'] = list(map(lambda x: model_account.classes_[x], np.argmax(model_account.predict_proba(test_data), axis = 1)))
    df_with_result.loc[:, 'probability_account'] = np.max(model_account.predict_proba(test_data), axis = 1)
    
    df_with_result.loc[:, 'pred_class_dedu'] = list(map(labmda x: model_dedu.classes_[x], np.argmax(model_dedu.predict_proba(test_data), axis = 1)))
    df_with_result.loc[:, 'probability_dedu'] = np.max(model_dedu.predict_proba(test_data), axis = 1)
    
    df_with_result = pd.concat([df_with_result, df_assign]).reset_index(drop = True)
    df_with_result.loc[df_with_result.pred_class_dedu.isnull(), 'pred_class_dedu'] = ''
    df_with_result.loc[df_with_result.probability_dedu.isnull(), 'probability_dedu'] = 0

    utils.insert_result(df_with_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type = str, default= None, help = 'Data when load data')
    args = parser.parse_args()

    predict(args.date)