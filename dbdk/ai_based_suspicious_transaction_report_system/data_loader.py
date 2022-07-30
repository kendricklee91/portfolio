import packages as pkgs
import utils

logger = utils.set_logger(__name__)

def data_load(data_path, today_dtm, bsn_dsc_code):
    logger.info('데이터 로딩 시작')
    
    # Connect to HDFS
    try:
        hdfs = pkgs.HDFileSystem(host = '', port = 8020, pars = {'' : 'kerberos'})
    except ConnectionError as e:
        print('연결 오류 :', e)
        hdfs = pkgs.HDFileSystem(host = '', port = 8020, pars = {'' : 'kerberos'})
    
    data = utils.read_table(hdfs, f"{data_path}/data_{bsn_dsc_code}_{today_dtm}", date_columns = ['tr_dt', 'anw_dt', 'anw_rg_dt'], is_hive = False)
    label = utils.read_table(hdfs, f"{data_path}/label_{bsn_dsc_code}_{today_dtm}", date_columns = ['tr_dt', 'dtc_dt'], is_hive = False)
    need_tr = utils.read_table(hdfs, f"{data_path}/need_tr_{bsn_dsc_code}_{today_dtm}", is_hive = False)
    
    target_cust = label['cusno'].unique()
    print(f"Target 고객 수 : {len(target_cust)}명")
    
    data_fe = data[data.tr_oc_bsn_dsc.isin(['FE'])].reset_index(drop = True) # 총 거래 및 당발, 타발 거래에 사용하는 DB
    data_nfe = data[~data.tr_ocu_bsn_dsc.isin(['FE'])].reset_index(drop = True) # 입출금 거래에 사용하는 DB    
    
    inherit_data = data[data['tr_trt_tpc'].isin(['R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9', 'RA', 'RH', 'RI'])].reset_index(drop = True) # 증여성송금 및 위험군 관련 거래에 사용하는 DB
    inherit_data = utils.drop_duplicate(inherit_data, 'intg_imps_key_val', [19, 20])
    
    logger.info('데이터 로딩 완료')
    return data, label, need_tr, data_fe, data_nfe, inherit_data, target_cust