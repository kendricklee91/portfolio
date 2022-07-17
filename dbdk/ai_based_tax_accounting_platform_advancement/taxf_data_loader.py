import taxf_config as config
import taxf_utils as utils
import pandas as pd
import datetime
import pymysql
import pickle
import time
import os

logger = utils.get_logger(__name__)

def data_loader():
    # 학습용 전체 데이터 불러오기

    start_time = time.time()
    
    conn = utils.connect() # DB Connect
    cursor = conn.cursor()

    query = """
        SELECT NO_BIZ, CD_SLIP, DT_TRAN, CD_TRAN, CD_STUS, CD_ELEC, NO_BIZ_C, TP_BIZ_C, CD_INDUSTRY_C, NM_INDUSTRY_C, NM_ITEM, AM_PRI, AM_VAT, AM_TOT, AM_FEE, AM_VAT_FARM, AM_SERVICE, CD_APPROVE, CD_ACCOUNT, CD_AI_STUS, CD_DEDU, MOD_NUM, CD_TAX \
        FROM AD_ACC_MASTER_backup \
        WHERE NOT CD_TRAN IN ('yessin2', 'bill1out', 'bill2out', 'bill4out', 'card4out', 'card5out', 'cash3out', 'cash4out', 'cash5out', 'home1out', 'home2out', 'home3out', 'leaseout')
    """
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    columns = [
        'NO_BIZ', 'CD_SLIP', 'DT_TRAN', 'CD_TRAN', 'CD_STUS', 'CD_ELEC', 'NO_BIZ_C', 'TP_BIZ_C', 'CD_INDUSTRY_C', 'NM_INDUSTRY_C', 'NM_ITEM',
        'AM_PRI', 'AM_VAT', 'AM_TOT', 'AM_FEE', 'AM_VAT_FARM', 'AM_SERVICE', 'CD_APPROVE', 'CD_ACCOUNT', 'CD_AI_STUS', 'CD_DEDU', 'MOD_NUM', 'CD_TAX'
    ]
    df = pd.DataFrame(list(data), columns = columns)

    end_time = time.time()

    process_time = end_time - start_time
    if process_time >= 60: # 데이터 load 처리 시간이 1분 이상일 때 (소수점 2째 자리까지 표현)
        logger.info(f"Data Loaded in {process_time/60:.2f}m")
    else:
        logger.info(f"Data Loaded in {process_time:.2f}s")

    if not os.path.exists(f"{config.DATA_PATH}"):
        os.mkdir(f"{config.DATA_PATH}")
    return df

def get_daily_data(date = None):
    # 일별 데이터 불러오기

    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    else:
        pass
    
    conn = utils.connect() # DB Connect
    cursor = conn.cursor()

    query = f"""
        SELECT NO_BIZ, CD_SLIP, DT_TRAN, CD_TRAN, CD_STUS, CD_ELEC, NO_BIZ_C, TP_BIZ_C, CD_INDUSTRY_C, NM_INDUSTRY_C, NM_ITEM, AM_PRI, AM_VAT, AM_TOT, AM_FEE, AM_VAT_FARM, AM_SERVICE, CD_APPROVE, CD_ACCOUNT, CD_AI_STUS, CD_DEDU, MOD_NUM, CD_TAX \
        FROM AD_ACC_MASTER \
        WHERE DT_REG LIKE '{date}%'
    """
    cursor.execute(query)
    date = cursor.fetchall()

    cursor.close()
    conn.close()

    columns = [
        'NO_BIZ', 'CD_SLIP', 'DT_TRAN', 'CD_TRAN', 'CD_STUS', 'CD_ELEC', 'NO_BIZ_C', 'TP_BIZ_C', 'CD_INDUSTRY_C', 'NM_INDUSTRY_C', 'NM_ITEM',
        'AM_PRI', 'AM_VAT', 'AM_TOT', 'AM_FEE', 'AM_VAT_FARM', 'AM_SERVICE', 'CD_APPROVE', 'CD_ACCOUNT', 'CD_AI_STUS', 'CD_DEDU', 'MOD_NUM', 'CD_TAX'
    ]
    df = pd.DataFrame(list(data), columns = columns)
    return df


    