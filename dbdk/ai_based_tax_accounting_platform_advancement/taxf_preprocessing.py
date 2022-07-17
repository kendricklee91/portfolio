from dbdeep.nlp import MorphologicalAnalyzer, SynonymHandler
from taxf_data_loader import data_loader
from collections import Counter
from tqdm import tqdm

import taxf_config as config
import taxf_utils as utils
import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
import time
import os
import re

logger = utils.get_logger(__name__)
warnings.filterwarnings('ignore')

def preprocessing(df:pd.DataFrame = None, inference:bool = False) -> pd.DataFrame:
    ################################
    ### Define private functions ###
    ################################
    def _filtering(df, inference = False):
        '''
        1차 데이터 필터링
        - Null 값 많은 변수 제거
        - CD_ACCOUNT(회계 계정)가 없거나 잘못된 CD_ACCOUNT에 대한 row 제거
        - Train 시 CD_ACCOUNT(회계 계정)의 수가 너무 적은 row 제거
        
        Parameters
        ----------
        df
        inference: bool
        
        Returns
        -------
        df
        '''

        if not inference: # not False -> True / not True -> False
            logger.info('Filtering Data')
            print(f"Number of rows before filtering : {df.shape[0]}")

            df = df.loc[df.CD_ACCOUNT != ''].reset_index(drop = True)
            df = df.loc[(df.CD_ACCOUNT != '') & ~(df.CD_ACCOUNT.map(account_code_mapper).isnull())].fillna('').reset_index(drop = True)
            drop_accounts = list(df.CD_ACCOUNT.value_counts().reset_index(name = 'counts').query('counts < 10').loc[: 'index'].values)
            df = df.query(f"CD_ACCOUNT not in {drop_accounts}").reset_index(drop = True)
            
            print(f"Number of rows after filtering : {df.shape[0]}")
            time.sleep(0.2)
        return df
    
    def _merge(df):
        '''
        df에 사용자정보 데이터 merge
        - 사용자정보 : 사업자번호, 회원사 업종, 회원사 업태

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        logger.info('Merge data')
        print(f"Shape of dataframe before merge : {df.shape}")

        df = df.merge(user_info, how = 'left', on = 'NO_BIZ')

        print(f"Shape of dataframe after merge : {df.shape}")
        time.sleep(0.2)
        return df
    
    def _create_variables(df):
        '''
        파생변수 생성

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        logger.info('Create derived variables')

        df.loc[:, 'NO_BIZ_sub'] = df.NO_BIZ.apply(lambda x: str(x)[3:5] if str(x) != '' else 100).astype(int)
        df.loc[:, 'NO_BIZ_C_sub'] = df.NO_BIZ_C.apply(lambda x: str(x)[3:5] if str(x) != '' else '').map(lambda x: int(x) if x != '' else 100)

        # 사업자번호의 4번째, 5번째 자리 값에 사업자번호 카테고리 매핑 적용
        df.loc[:, 'NO_BIZ_CAT'] = df.NO_BIZ_sub.map(no_biz_mapper)
        df.loc[:, 'NO_BIZ_C_CAT'] = df.NO_BIZ_C_sub.map(no_biz_mapper).fillna(0).astype(int)
        df.loc[:, 'TRAN'] = df.NO_BIZ_CAT.astype('str') + df.NO_BIZ_C_CAT.astype('str')
        
        # NO_BIZ_CAT : 사업자번호(카테고리), NO_BIZ_C_CAT : 거래처사업자번호(카테고리), TRAN : NO_BIZ_CAT과 NO_BIZ_C_CAT 간 거래 관계
        print('Create NO_BIZ_CAT, NO_BIZ_C_CAT, TRAN derived variables')
        time.sleep(0.2)
        return df
    
    def _treat_missing_values(df):
        '''
        결측치 ('') 처리
        - 결측치 값을 0으로 대체
        - 결측치 처리 변수
            - CD_SLIP(전표번호), TP_BIZ_C(거래처 가맹점 분류), CD_TAX(과세구분코드), DT_TRAN(거래일자)

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        logger.info('Treat missing values')

        df.loc[:, 'CD_SLIP'] = np.where(df.CD_SLIP == '', 0, df.CD_SLIP) #
        df.loc[:, 'TP_BIZ_C'] = np.where(df.TP_BIZ_C == '', 0, df.TP_BIZ_C)
        df.loc[:, 'CD_TAX'] = np.where(df.CD_TAX == '', 0, df.CD_TAX)
        df.loc[:, 'DT_TRAN'] = df.DT_TRAN.apply(lambda x: x[-2:]).astype(int)
        return df
    
    def _treat_oulier_words(df):
        '''
        한글 값을 가진 변수들의 특수문자 처리
        - 특수문자 처리 변수
            - CD_INDUSTRY_C(거래처 업종 코드), NM_INDUSTRY_C(거래처 업태), NM_INDUSTRY(회원사 업종), NM_INDUSTRY_SUB(회원사 업태)

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        logger.info('Special word processing')

        for col in text_cols:
            try:
                for val, val_repl in repl.items():
                    df.loc[:, col] = df.loc[:, col].str.replace(val, val_repl)
            except Exception as e:
                print(f"{col} : {e}")
        return df
    
    def _concat_dummies(df):
        '''
        Dummy화 변수 처리 및 concat
        - Dummy화 변수
            - CD_SLIP(전표번호), CD_TRAN(거래유형), TP_BIZ_C(거래처 가맹점 분류), CD_AI_STUS(인공지능 처리상태), MOD_NUM(수정횟수), CD_TAX(과세구분코드),
            - NO_BIZ_CAT(사업자번호 카테고리), NO_BIZ_C_CAT(거래처 사업자번호 카테고리), TRAN(NO_BIZ_CAT과 NO_BIZ_C_CAT 간 거래 관계)

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        logger.info('Create dummy variables and concat')

        for col in dummy_cols:
            dummy_df = pd.get_dummies(df.loc[:, col], prefix = col)
            df = df.drop(col, axis = 1)
            df = pd.concat([df, dummy_df], axis = 1)
        return df
    
    def _synonym_handling(df):
        '''
        업종, 업태 유의어 처리 및 one-hot encoding
        변수 dummy화 및 concat

        - Dummy화 변수
            - CD_INDUSTRY_C(거래처 업종 코드), NM_INDUSTRY_C(거래처 업태), NM_INDUSTRY(회원사 업종), NM_INDUSTRY_SUB(회원사 업태)

        Parameters
        ----------
        df
        
        Returns
        -------
        df
        '''

        sh = SynonymHandler()

        for col in text_cols:
            try:
                concat_df = sh.synonym_handler(df, col)
                df = pd.concat([df, concat_df], axis = 1)
            except FileNotFoundError:
                utils.set_synonym_dict()
        return df
    
    def _nm_item_handling(df, inference:bool = False):
        '''
        품목 형태소 분석(명사) 및 one-hot encoding
        변수 dummy화 및 concat

        - Dummy화 변수
            - CD_INDUSTRY_C(거래처 업종 코드), NM_INDUSTRY_C(거래처 업태), NM_INDUSTRY(회원사 업종), NM_INDUSTRY_SUB(회원사 업태)

        Parameters
        ----------
        df
        inference

        Returns
        -------
        df
        '''

        ma = MorphologicalAnalyzer()

        ma_result = []
        ma_dict = {}
        for val in tqdm(df.loc[(df.NM_ITEM != '') & ~(df.NM_ITEM.isnull()). 'NM_ITEM'].values):
            res = ma.parse(val)

            tmp_list = []
            for data in res:
                if data[1].startswith('N'):
                    ma_result.append(data[0])
                    tmp_list.append(data[0])
            ma_dict[val] = tmp_list
        
        if not inference: # not False -> True / not True -> False
            count_word = Counter(ma_result)
            count_word = {key : val for key, val in sorted(dict(count_word).items(), key = lambda x: x[1], reverse = True)}

            word_freq = config.WORD_FREQ
            using_words = list({key : val for key, val in count_word.items() if val > word_freq}.keys())
            using_words = list(pd.Series(list(map(_filter_words, using_words))).dropna().values)
            joblib.dump(using_words, f"{config.DATA_PATH}/using_words_list.pkl")
        else: # not True -> False / not False -> True
            using_words = joblib.load(f"{config.DATA_PATH}/using_words_list.pkl")
        
        concat_df = pd.get_dummies(df.loc[(df.NM_ITEM != '') & ~(df.NM_ITEM.isnull()), 'NM_ITEM'].map(ma_dict).explode().apply(lambda x: x if x in using_words else np.nan), prefix = 'NM_ITEM').sum(level = 0)
        df = pd.concat([df, concat_df], axis = 1).fillna(0)
        return df
    
    def _filter_words(word):
        word = word.strip('[]/<>')
        word = re.sub('/', '', word)

        if '' == word:
            return np.nan
        return word
    
    def _drop_cols(df):
        df = df.drop(text_cols, axis = 1)
        df = df.drop(drop_cols, axis = 1)
        df = df.fillna(0)
        return df
    
    ###############################
    ### Define public functions ###
    ###############################

    def check_data(df):
        '''
        제거할 변수이거나 추후 다룰 변수가 아닌 변수들 중 숫자형이 아닌 값을 가지고 있는 변수가 있는지 체크

        - 체크할 변수
            - NM_ITEM(품목), CD_CLIP(전표번호), TP_BIZ_C(거래처 가맹점 분류), CD_TAX(과세구분코드)
            - CD_INDUSTRY_C(거래처 업종 코드), NM_INDUSTRY_C(거래처 업태), CD_DEDU(공제여부), CD_ACCOUNT(계정과목)

        Parameters
        ----------
        df

        Returns
        -------
        df
        '''

        logger.info('Check data')

        err_count = 0
        err_col = []
        for col in [col for col in df.columns if col not drop_cols + later_cols]:
            try:
                df.loc[:, col].astype(int)
            except Exception as e:
                err_col.append(col)
                if df.loc[df[col] == ''].shape[0] > 0:
                    err_count += 1
                    print(f"{col} have non-numerical values and some black values")
        
        if 0 == err_count:
            return True
        else:
            return False

    ################################
    ###    Start Preprocessing   ###
    ################################

    if df is None:
        with open(f"{config.DATA_PATH}/master.pkl", 'rb') as f:
            df = pickle.load(f)
    else:
        df = df

    text_cols  = ['CD_INDUSTRY_C', 'NM_INDUSTRY_C', 'NM_INDUSTRY', 'NM_INDUSTRY_SUB']
    dummy_cols = ['CD_SLIP', 'CD_TRAN', 'TP_BIZ_C', 'CD_AI_STUS', 'MOD_NUM', 'CD_TAX', 'NO_BIZ_CAT', 'NO_BIZ_C_CAT', 'TRAN']
    drop_cols  = ['NO_BIZ_sub', 'NO_BIZ_C_sub', 'NO_BIZ','DT_TRAN', 'NO_BIZ_C', 'CD_STUS', 'NM_ITEM']
    later_cols = ['NM_ITEM', 'CD_SLIP', 'TP_BIZ_C', 'CD_TAX', 'CD_INDUSTRY_C', 'NM_INDUSTRY_C', 'CD_DEDU', 'CD_ACCOUNT']

    repl = config.REPLACE # 단어 치환 정보 (dictionary type)
    user_info = utils.get_user_info() # 사용자 정보 가져오기 (사업자번호, 회원사 업종, 회원사 업태)

    if not os.path.exists(f"{config.DATA_PATH}/no_biz_mapper.pkl"):
        utils.set_no_biz_mapper()
    no_biz_mapper = utils.get_no_biz_mapper() # 사업자번호 별 카테고리 dictionary
    account_code_mapper = utils.get_account_code_mapper() # 계정과목코드와 계정과목명 매칭 dictionary

    check_data = check_data(df)
    if check_data:
        logger.info('Start Preprocessing')

        df = _filtering(df, inference = inference)
        df = _merge(df)
        df = _create_variables(df)
        df = _treat_missing_values(df)
        df = _treat_oulier_words(df)
        df = _concat_dummies(df)
        df = _synonym_handling(df)

        using_NM_ITEM = config.USING_NM_ITEM
        if using_NM_ITEM:
            df = _nm_item_handling(df, inference = inference)
        df = _drop_cols(df)

        if not inference: # not False -> True / not True -> False
            if os.path.exists(f"{config.DATA_PATH}/concat_in_prediction.pkl"):
                os.rename(f"{config.DATA_PATH}/concat_in_prediction.pkl", f"{config.DATA_PATH}/concat_in_prediction_backup.pkl")
            joblib.dumb(df.drop(['CD_ACCOUNT', 'CD_DEDU'], axis = 1).head(0), f"{config.DATA_PATH}/concat_in_prediction.pkl")
        else: # not True -> False / not False -> True
            data_for_concat = joblib.load(f"{config.DATA_PATH}/concat_in_prediction.pkl")
            col_shape = data_for_concat.shape[1]

            df = pd.concat([data_for_concat, df]).fillna(0)
            df = df.iloc[:, :col_shape]
        return df
    else:
        logger.info('Check data first')