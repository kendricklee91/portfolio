import packages as pkgs

# Variable initialization
TODAY_DTM = pkgs.datetime.now().strftime('%Y-%m-%d')
FILE_NAME = 'forex' # deposit (수신) / forex (외환)
BSN_DSC_CODE = '05' # 01 (수신) / 05 (외환)

# Directory paths
MODEL_PATH = f'/tmp/ai-str/xgb_model.model'

DATA_PATH = f'/tmp/ai-str/temp'
HDFS_PATH = f'/tmp/ai-str/temp/label_{BSN_DSC_CODE}'
HDFS_TABLE_PATH = f'/apps/hive/warehouse/nbbpmt.db/tb_ml_bk_sh_ai_lrn_data_fx'

RESULT_LOCAL_PATH = f'/home/work/modeling/result/tmp'
LOG_SAVE_PATH = f'/home/work/modeling/log/preprocessing/bsn_dsc_{BSN_DSC_CODE}/'

# File name variables
RESULT_FILE_NAME = f'{FILE_NAME}_{TODAY_DTM}.parquet'
LOG_FILE_NAME = f'{FILE_NAME}_{TODAY_DTM}.log'
HDFS_FILE_NAME = f'{TODAY_DTM}.parquet'

# Columns related to load tables
RZT_COLUMNS = ['cusno', 'tr_dt', 'sspn_tr_rule_id', 'sspn_tr_stsc', 'intg_imps_key_val', 'dcz_sqno']
DCZ_COLUMNS = ['dcz_sqno', 'now_sts_dsc', 'dcz_rqr_dt']
CM_CUST_COLUMNS = ['cusno', 'rep_cusno', 'cus_tpc', 'bzcc', 'rnm_dsc', 'dmd_dptr_accn', 'svtp_dptr_accn', 'ts_tr_acn', 'gen_la_tr_acn',
                   'bildc_tr_acn', 'acgrn_tr_acn', 'fx_tr_acn', 'mad_tr_acn', 'cd_tr_acn', 'cus_job_cfc', 'ag', 'anw_rg_dt']

# Variables related to creation of derived variables
DEPOSIT_CODE = ['01', '04', '08', '12', '14']                             # 입금거래
PAYMENT_CODE = ['02', '05', '09', '13', '15']                             # 출금거래 - Payment
WITHDRAW_CODE = ['02']                                                    # 출금거래 - Withdraw
NO_FE_CODE = ['486', '487']                                               # 비외환거래
NFTF_CODE = ['02', '04', '05']                                            # 비대면거래
EXCH_CODE = ['01']                                                        # 환전거래
OUTWARD_CODE = ['486']                                                    # 당발거래
INWARD_CODE = ['487']                                                     # 타발거래
INHERIT_ALL_CODE = ['R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9', 'RA'] # 증여성거래
INHERIT_OUTWARD_CODE = ['R1', 'R2']                                       # 증여성송금거래
INHERIT_INWARD_CODE = ['R3', 'R4']                                        # 증여성타발거래
INHERIT_TERROR_CODE = ['R6', 'R7', 'R8', 'R9', 'RA']                      # 증여성테러단체거래
HIGH_RISK_CONUTRY_CODE = ['R6', 'R7', 'R8', 'R9', 'RA', 'RH', 'RI']       # 위험국 관련 거래

# Preprocessing parameters

# Model parameters
TRAIN_IN_VALID_SIZE = 0.1 # train data 내에서 생성할 검증 데이터의 사이즈 크기
RANDOM_SEED = 34

TRAIN_START_DATE = '2020-06-01' # 생성할 학습 데이터의 시작 날짜
TRAIN_END_DATE = '2021-05-31' # 생성할 학습 데이터의 마지막 날짜

VALIDATION_START_DATE = '2021-05-15' # train data 내에서 생성할 검증 데이터의 시작 날짜

DTC_START_DATE = '2021-05-01' # 실제 Alert(검출)이 된 데이터의 시작 날짜
DTC_END_DATE = '2021-06-30' # 실제 Alert(검출)이 된 데이터의 마지막 날짜

SUSPICIOUS_IMBALANCE_SAMPLING = False # 혐의 / 비혐의 별 기간을 다르게 설정 여부
TRAIN_VALIDATION_SPLIT_BY_PERIOD = False # 기간에 따라 학습 / 검증셋 분리 여부
SHAPELY_RESULT_SAVE = False # Shapely 결과 저장 여부

NH_AML_VARIABLE_DEF_FILE_PATH = './fault_eda/data/var_list.csv' # 외화 관련 feature 정보 (feature 명, 고객유형, old_fillna_val, new_fillna_val, feature type, 한글 feature 명, description)
NM_AML_VARIABLE_FILTER_FILE_PATH = './fault_eda/data/var_filter_list.csv' # filter 적용하지 않을 feature (filter : test data에서 filter 적용할 컬럼의 특정값 변경)

HYPERPARAMETER_OPT = True
HYPERPARAMETER_OPT_ITER = 10
HYPERPARAMETER_OPT_UPDATE = True

# Hyper-parameters bounds
PBOUNDS = {
    'max_depth'        : (5, 10),
    'learning_rate'    : (0.01, 0.3),
    'n_estimators'     : (50, 1000),
    'gamma'            : (0.001, 0.01),
    'min_child_weight' : (2, 10),
    'subsample'        : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'alpha'            : (0.0, 0.5),
    'reg_alpha'        : (0, 2),
    'reg_lambda'       : (0, 2)
}

ROOT_PATH = '/tmp/ai-str/02/test_result'
NOW_DT = datetime.today().strftime('%Y%m%d')
NOW_DTM = datetime.today().strftime('%Y%m%d_%H%M%S')

CHANGE_COLS_TO_BOOLEAN = [
    'F_Fn_Yn',                 # 외국인 여부
    'F_Npc_Yn',                # 비영리 법인 여부
    'F_Rg_Tr_Yn',              # 위험군 관련 거래 이력 보유 여부
    'F_Cust_Risk_Job_Yn',      # 고위험 직업 여부
    'F_1D_Cust_Terror_Tr_Yn',  # 당일 테러국 관련 거래 여부
    'F_1D_Cust_Sanction_Tr_Yn' # 당일 제재국 및 기타 위험국 관련 거래 여부
]

CHANGE_COLS_DUMMY_TYPE_COMBINE = [
    'cus_tpc',                   # 개인 / 법인 구분 코드
    'F_1D_Tr_Time_Type',         # 고객 별 거래 건당 거래 시간 유형 (오전, 오후, 업무 외)
    'F_Svc_Type',                # 고객 별 이용 서비스 정보
    'F_7D_Cust_Fc_Dep_Intensity' # 최근 7일 동안 입금 강도 (최근 7일 동안 출금거래 누계액 / 입금거래 누계액)
]

RESULT_META_COLS = ['cusno', 'bsn_dsc', 'tr_dt', 'dtc_dt', 'suspicious', 'predict', '0_proba', '1_proba'] # 분류 결과 관련 필요 COLUMNS