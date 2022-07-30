from processing import preprocess
from data_loader import data_load
from modeling import model

import packages as pkgs
import config as cfg

data_path = cfg.DATA_PATH
today_dtm = cfg.TODAY_DTM
bsn_dsc_code = cfg.BSN_DSC_CODE

# Load data
data, label, need_tr, data_fe, data_nfe, inherit_data, target_cust = data_load(date_path, today_dtm, bsn_dsc_code)

# Preprocess
try:
    hdfs = pkgs.HDFileSystem(host = '', port = 8020, pars = {'' : 'kerberos'})
except ConnectionError as e:
    hdfs = pkgs.HDFileSystem(host = '', port = 8020, pars = {'' : 'kerberos'})
preprocessed_df = preprocess(hdfs, data, label, need_tr, data_fe, data_nfe, inherit_data, target_cust)

# Data pipeline
dp = utils.DATA_PIPELINE()
total_data = dp.add_data_frame({'total_data' : preprocessed_df}).data_combine().type_chage_final(cfg.CHANGE_COLS_TO_BOOLEAN, 'uint8').generate_dummy(cfg.CHANGE_COLS_DUMMY_TYPE_COMBINE).show_type('total_data').get_total_frame()

# Model train, test performance of trained model
metric_df, confusion_df, shapely_top5_df = model(total_data, case = 1)

print(metric_df)
print(confusion_df)
print(shapely_top5_df)