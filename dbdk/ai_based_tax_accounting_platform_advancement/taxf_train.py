from taxf_preprocessing import preprocessing
from taxf_data_loader import data_loader
from taxf_modeling import Model

import taxf_config as config
import taxf_utils as utils

def train():
    df = data_loader()     # 학습용 전체 데이터 불러오기
    df = preprocessing(df) # 학습용 전체 데이터 전처리 진행

    # 회계 계정 과목
    model_account = Model(df) # 모델 생성

    if config.TUNING:
        model_acocunt.param_tuning()
    
    model_account.train()
    model_account.thresh_per_cls()

    # 회계 공제 여부
    model_dedu = Model(df, target = 'CD_DECU')

    if config.TUNING:
        model_dedu.param_tuning()
    
    model_dedu.train()
    model_dedu.thresh_per_cls()

if __name__ = '__main__':
    train()