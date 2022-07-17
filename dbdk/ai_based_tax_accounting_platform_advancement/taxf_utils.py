import taxf_config as config

import pandas as pd
import numpy as np
import logging
import pymysql
import pickle
import time
import json

_log_format = f'[%(asctime)s] %(name)s [%(levelname)s] : %(message)s'

def connect():
    # 세친구 DB 서버 접속
    conn = pymysql.connect(user = 'dbdk', passwd = 'dbdk027857627', host = '49.50.162.35', port = 3306, database = 'taxpaldb', charset = 'utf8', connect_timeout = 300)
    logger.info('DB Connect')
    return conn

def get_all_table_description():
    # 전체 테이블 개요

    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name, table_rows, table_comment FROM information_schema.tables WHERE table_schema = 'taxpaldb'")

    _data =cursor.fetchall()

    names = []
    rows = []
    comments = []
    for name, row, comment in _data:
        if name.startswith('v_'):
            continue
        names.append(name)
        rows.append(row)
        comments.append(comment)
    tagle_description = pd.DataFrame({'TABLE_NAME' : names, 'TABLE_COMMENT' : comments, 'TABLE_ROWS' : rows})

    cursor.close()
    conn.close()
    return table_description

def get_columns(table_name):
    # table_name의 컬럼명 가져오기

    conn = connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'taxpaldb'")

    data = cursor.fetchall()
    columns = [col for table, col in data in table == table_name]
    
    cursor.close()
    conn.close()
    return columns

def get_account_code_mapper():
    # 계정과목코드와 계정과목명을 매칭해주는 dictionary 생성

    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT CD_ACCOUNT, NM_ACCOUNT FROM AD_ACCOUNT_CODE')

    data = cursor.fetchall()
    account_code_dict = {key : val for key, val in data}

    cursor.close()
    conn.close()
    return account_code_dict

def get_user_info():
    # 사용자 정보 가져오기

    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT NO_BIZ, NM_INDUSTRY, NM_INDUSTRY_SUB FROM USER_INFO WHERE NOT NM_INDUSTRY IS NULL AND NM_INDUSTRY != ''")

    data = cursor.fetchall()
    user_info = pd.DataFrame(data, columns = ['NO_BIZ', 'NM_INDUSTRY', 'NM_INDUSTRY_SUB']) # 사업자번호, 회원사 업종, 회원사 업태

    cursor.close()
    conn.close()
    return user_info

def set_synonym_dict():
    # 유의어 처리용 dictionary 생성

    repl = {
        '�' : '',
        '\u3000' : '',
        '，' : ',',
        '・' : ',',
        'ㆍ' : ',',
        '·' : ',',
        '．' : ',',
        '.' : ',',
        '\+' : ''
    }

    CD_INDUSTRY_C = pd.read_excel(f"{config.DATA_PATH}/DBDK_유의어처리.xlsx", sheet_name = 'CD_INDUSTRY_C')
    CD_INDUSTRY_C = CD_INDUSTRY_C.fillna('') # NaN -> 공백 처리

    # Excel 시트에 있는 각 컬럼명을 변경 후 concat 
    CD_INDUSTRY_C1 = pd.concat([
        CD_INDUSTRY_C.iloc[:,1:3].rename(columns = {'업종1' : 'col1', '구분1' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,3:5].rename(columns = {'업종2' : 'col1', '구분2' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,5:7].rename(columns = {'업종3' : 'col1', '구분3' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,7:9].rename(columns = {'업종4' : 'col1', '구분4' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,9:11].rename(columns = {'업종5' : 'col1', '구분5' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,11:13].rename(columns = {'업종6' : 'col1', '구분6' : 'col2'}),
        CD_INDUSTRY_C.iloc[:,13:15].rename(columns = {'업종7' : 'col1', '구분7' : 'col2'})
    ])

    # df에서 col1이 값이 있는 부분만 가져오고 col1을 index로 설정한 후 dictionary 형태로
    # 변경해서 키가 col2인 value를 가져오기 col2의 밸류 형식 - key:val
    CD_INDUSTRY_C_dict = CD_INDUSTRY_C1.loc[CD_INDUSTRY_C1.col1 != ''].set_index('col1').to_dict().get('col2')

    NM_INDUSTRY_C = pd.read_excel(f"{config.DATA_PATH}/DBDK_유의어처리.xlsx", sheet_name = 'NM_INDUSTRY_C')
    for val, val_repl in repl.items():
        NM_INDUSTRY_C.loc[:, '거래처_업태'] = NM_INDUSTRY_C['거래처_업태'].str.replace(val, val_repl)
    NM_INDUSTRY_C_dict = NM_INDUSTRY_C.set_index("거래처_업태").to_dict().get('구분1')

    NM_INDUSTRY = pd.read_excel(f"{config.DATA_PATH}/DBDK_유의어처리.xlsx", sheet_name = '회원정보리스트')
    NM_INDUSTRY = NM_INDUSTRY.loc[:,['분류1', '분류2', '업태', '종목']].fillna('')
    for val, val_repl in repl.items():
        NM_INDUSTRY.loc[:, '업태'] = NM_INDUSTRY['업태'].str.replace(val, val_repl)
    for val, val_repl in repl.items():
        NM_INDUSTRY.loc[:, '종목'] = NM_INDUSTRY['종목'].str.replace(val, val_repl)
    NM_INDUSTRY_dict = NM_INDUSTRY.loc[NM_INDUSTRY['업태'] != '', ['분류1', '업태']].set_index('업태').to_dict().get('분류1')
    NM_INDUSTRY_SUB_dict = NM_INDUSTRY.loc[NM_INDUSTRY['종목'] != '',['분류2', '종목']].set_index('종목').to_dict().get('분류2')

    with open(f'{config.DATA_PATH}/CD_INDUSTRY_C_dict.json', 'w') as f:
        json.dump(CD_INDUSTRY_C_dict, f)
        
    with open(f'{config.DATA_PATH}/NM_INDUSTRY_C_dict.json', 'w') as f:
        json.dump(NM_INDUSTRY_C_dict, f)

    with open(f'{config.DATA_PATH}/NM_INDUSTRY_dict.json', 'w') as f:
        json.dump(NM_INDUSTRY_dict, f)

    with open(f'{config.DATA_PATH}/NM_INDUSTRY_SUB_dict.json', 'w') as f:
        json.dump(NM_INDUSTRY_SUB_dict, f)

def set_no_biz_mapper():
    # 사업자번호 별 카테고리 dictionary 생성

    NO_BIZ_MAPPER = {k : 1 for k in range(1, 80)} # 0 ~ 79의 key에 val을 초기화 값으로 1로 설정

    # dictionary key 추가 생성
    NO_BIZ_MAPPER[80] = 2
    NO_BIZ_MAPPER[81] = 3
    NO_BIZ_MAPPER[82] = 4
    NO_BIZ_MAPPER[83] = 5
    NO_BIZ_MAPPER[84] = 6
    NO_BIZ_MAPPER[85] = 7
    NO_BIZ_MAPPER[86] = 3
    NO_BIZ_MAPPER[87] = 3
    NO_BIZ_MAPPER[89] = 8

    # dictionary key 추가 생성 (90 ~ 99), val 초기화 값은 9
    for i in range(90, 100):
        NO_BIZ_MAPPER[i] = 9
    
    # 객체를 pickle로 저장할 때, 메모리가 부족해져 나머지 작업 실행이 중단되고 파일이 제대로 저장하지 않을 때 HIGHEST_PROTOCOL을 사용해 정상적으로 저장 가능하게 설정
    with open(f"{config.DATA_PATH}/no_biz_mapper.pkl", 'wb') as f:
        pickle.dump(NO_BIZ_MAPPER, f, protocol = pickle.HIGHEST_PROTOCOL) 

def get_no_biz_mapper():
    # 사업자번호 별 카테고리 dictionary를 불러 옴
    with open(f"{config.DATA_PATH}/no_biz_mapper.pkl", 'rb') as f:
        no_biz_mapper = pickle.load(f)
    return no_biz_mapper

def insert_result(result):
    # Inference 결과를 DB에 저장

    conn = connect()
    cursor = conn.cursor()

    _result = list(map(tuple, result.values))

    query = """
        INSERT INTO AD_ACC_MASTER_RESULT(NO_BIZ, CD_SLIP, DT_TRAN, CD_TRAN, CD_STUS, CD_ELEC, NO_BIZ_C, TP_BIZ_C, CD_INDUSTRY_C, NM_INDUSTRY_C, NM_ITEM, AM_PRI, AM_VAT, AM_TOT, AM_FEE, AM_VAT_FARM, AM_SERVICE, CD_APPROVE, CD_ACCOUNT, CD_AI_STUS, CD_DEDU, MOD_NUM, CD_TAX, PRED_CLASS, PROBABILITY, PRED_CLASS_DEDU, PROBABILITY_DEDU)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(query, _result)

    conn.commit()
    conn.close()
    
def get_fileHandler():
    fileHandler = logging.FileHandler(f"{config.LOG_PATH}/log.txt")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logging.Formatter(_log_format))
    return fileHandler

def get_streamHandler():
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(logging.Formatter(_log_format))
    return streamHandler

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_fileHandler())
    logger.addHandler(get_streamHandler())
    return logger

logger = get_logger(__name__)