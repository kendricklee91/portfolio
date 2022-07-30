import packages as pkgs
import config as cfg
import utils

logger = utils.set_logger(__name__)

def preprocess(hdfs, data, label, need_tr, data_fe, data_nfe, inherit_data, target_cust):
    logger.info('파생변수 생성')
    
    logger.info('\t RZT 데이터 로딩')
    rzt_data = utils.read_table(hdfs, 'tb_ml_bk_sh_xtr_rzt', columns = cfg.RZT_COLUMNS,
                                data_types = {'cusno' : str, 'sspn_tr_stsc' : 'category', 'dcz_sqno' : int}, date_columns = ['tr_dt'], target_cust = target_cust)
    
    logger.info('\t DCZ 데이터 로딩')
    dcz_data = utils.read_table(hdfs, 'tb_ml_bk_cm_dcz', columns = cfg.DCZ_COLUMNS,
                                data_types = {'dcz_sqno' : str, 'dcz_rqr_dt' : str, 'now_sts_dsc' : 'category'}, date_columns = ['dcz_rqr_dt'])
    
    logger.info('\t CM_CUST 데이터 로딩')
    cm_cust_data = utils.read_table(hdfs, 'tb_ml_bk_cm_cust', columns = cfg.CM_CUST_COLUMNS,
                                    data_types = {'cusno' : str, 'bzcc' : str, 'rnm_dsc' : 'category', 'rep_cusno' : str, 'cus_tpc' : 'category', 'ag' : int, 'dmd_dptr_accn' : int, 'svtp_dptr_accn' : int,
                                                  'ts_tr_acn' : int, 'gen_la_tr_acn' : int, 'bildc_tr_acn' : int, 'acgrn_tr_acn' : int, 'fx_tr_acn' : int, 'mad_tr_acn' : int, 'cd_tr_acn' : int},
                                    date_columns = ['anw_rg_dt'])
    
    logger.info('\t Part. 1')
    logger.info('\t 1. 입금거래')
    deposit_tr = utils.filter_tr_data(data_nfe, [f"cptld_tr_kdc in {cfg.DEPOSIT_CODE}"])
    if 0 != deposit_tr.shape[0]:
        deposit_tr['cnt'] = 1
        deposit_tr = utils.utils.reindex_base_table(deposit_tr, label, fill = True, fill_col = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 1.1. 입금거래 60일 Sum / Avg / Std / Count')
        deposit_60d_add = utils.utils.add_rolling_feature(deposit_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '61D',
                                                          var_names = ['F_60D_Cust_Dep_Sum', 'F_60D_Cust_Dep_Avg', 'F_60D_Cust_Dep_Std', 'F_60D_Cust_Dep_Cnt'])
        label = utils.utils.append_feature(label, deposit_60d_add, fill_value = {'F_60D_Cust_Dep_Sum' : -1, 'F_60D_Cust_Dep_Avg' : -1, 'F_60D_Cust_Dep_Std' : -1, 'F_60D_Cust_Dep_Cnt' : 0}, dtypes = {'F_60D_Cust_Dep_Cnt' : int})
        
        logger.info('\t 1.2. 입금거래 30일 Sum / Avg / Std / Count')
        deposit_30d_add = utils.utils.add_rolling_feature(deposit_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '31D',
                                                          var_names = ['F_30D_Cust_Dep_Sum', 'F_30D_Cust_Dep_Max', 'F_30D_Cust_Dep_Std', 'F_30D_Cust_Dep_Cnt'])
        label = utils.utils.append_feature(label, deposit_30d_add, fill_value = {'F_30D_Cust_Dep_Sum' : -1, 'F_30D_Cust_Dep_Avg' : -1, 'F_30D_Cust_Dep_Std' : -1, 'F_30D_Cust_Dep_Cnt' : 0}, dtypes = {'F_30D_Cust_Dep_Cnt' : int})
        
        logger.info('\t 1.3. 입금거래 15일 Sum / Avg / Std / Count')
        deposit_15d_add = utils.utils.add_rolling_feature(deposit_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '16D',
                                                          var_names = ['F_15D_Cust_Dep_Sum', 'F_15D_Cust_Dep_Avg', 'F_15D_Cust_Dep_Std', 'F_15D_Cust_Dep_Cnt'])
        label = utils.utils.append_feature(label, deposit_15d_add, fill_value = {'F_15D_Cust_Dep_Sum' : -1, 'F_15D_Cust_Dep_Avg' : -1, 'F_15D_Cust_Dep_Std' : -1, 'F_15D_Cust_Dep_Cnt' : 0}, dtypes = {'F_15D_Cust_Dep_Cnt' : int})
    else:
        logger.info('\t 1.1. 입금거래 60일 Sum / Avg / Std / Count')
        label['F_60D_Cust_Dep_Sum'] = float(-1)
        label['F_60D_Cust_Dep_Avg'] = float(-1)
        label['F_60D_Cust_Dep_Std'] = float(-1)
        label['F_60D_Cust_Dep_Cnt'] = 0
        
        label['F_60D_Cust_Dep_Sum'] = label['F_60D_Cust_Dep_Sum'].astyp(float)
        label['F_60D_Cust_Dep_Avg'] = label['F_60D_Cust_Dep_Avg'].astyp(float)
        label['F_60D_Cust_Dep_Std'] = label['F_60D_Cust_Dep_Std'].astyp(float)
        label['F_60D_Cust_Dep_Cnt'] = label['F_60D_Cust_Dep_Cnt'].astype(int)
        
        logger.info('\t 1.2. 입금거래 30일 Sum / Avg / Std / Count')
        label['F_30D_Cust_Dep_Sum'] = float(-1)
        label['F_30D_Cust_Dep_Avg'] = float(-1)
        label['F_30D_Cust_Dep_Std'] = float(-1)
        label['F_30D_Cust_Dep_Cnt'] = 0
        
        label['F_30D_Cust_Dep_Sum'] = label['F_30D_Cust_Dep_Sum'].astype(float)
        label['F_30D_Cust_Dep_Avg'] = label['F_30D_Cust_Dep_Avg'].astype(float)
        label['F_30D_Cust_Dep_Std'] = label['F_30D_Cust_Dep_Std'].astype(float)
        label['F_30D_Cust_Dep_Cnt'] = label['F_30D_Cust_Dep_Cnt'].astype(int)
                
        logger.info('\t 1.3. 입금거래 15일 Sum / Avg / Std / Count')
        label['F_15D_Cust_Dep_Sum'] = float(-1)
        label['F_15D_Cust_Dep_Avg'] = float(-1)
        label['F_15D_Cust_Dep_Std'] = float(-1)
        label['F_15D_Cust_Dep_Cnt'] = 0
        
        label['F_15D_Cust_Dep_Sum'] = label['F_15D_Cust_Dep_Sum'].astyp(float)
        label['F_15D_Cust_Dep_Avg'] = label['F_15D_Cust_Dep_Avg'].astyp(float)
        label['F_15D_Cust_Dep_Std'] = label['F_15D_Cust_Dep_Std'].astyp(float)
        label['F_15D_Cust_Dep_Cnt'] = label['F_15D_Cust_Dep_Cnt'].astype(int)
    del deposit_tr
    
    logger.info('\t 2. 출금거래 - payment')
    payment_tr = utils.filter_tr_data(data_nfe, [f"cptld_tr_kdc in {cfg.PAYMENT_CODE}"])    
    if 0 != payment_tr.shape[0]:
        payment_tr['cnt'] = 1
        payment_tr = utils.utils.reindex_base_table(payment_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 2.1. 출금거래 60일 Sum / Avg / Std / Count')
        payment_60d_add = utils.utils.add_rolling_feature(payment_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '61D',
                                                          var_names = ['F_60D_Cust_Pymt_Sum', 'F_60D_Cust_Pymt_Avg', 'F_60D_Cust_Pymt_Std', 'F_60D_Cust_Pymt_Cnt'])
        label = utils.utils.append_feature(label, payment_60d_add, fill_value = {'F_60D_Cust_Pymt_Sum' : -1, 'F_60D_Cust_Pymt_Avg' : -1, 'F_60D_Cust_Pymt_Std' : -1, 'F_60D_Cust_Pymt_Cnt' : 0}, dtypes = {'F_60D_Cust_Pymt_Cnt' : int})
        
        logger.info('\t 2.2. 출금거래 30일 Sum / Avg / Std / Count')
        payment_30d_add = utils.utils.add_rolling_feature(payment_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '31D',
                                                          var_names = ['F_30D_Cust_Pymt_Sum', 'F_30D_Cust_Pymt_Avg', 'F_30D_Cust_Pymt_Std', 'F_30D_Cust_Pymt_Cnt'])
        label = utils.utils.append_feature(label, payment_30d_add, fill_value = {'F_30D_Cust_Pymt_Sum' : -1, 'F_30D_Cust_Pymt_Avg' : -1, 'F_30D_Cust_Pymt_Std' : -1, 'F_30D_Cust_Pymt_Cnt' : 0}, dtypes = {'F_30D_Cust_Pymt_Cnt' : int})
        
        logger.info('\t 2.3. 출금거래 15일 Sum / Avg / Std / Count')
        payment_15d_add = utils.utils.add_rolling_feature(payment_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '16D',
                                                          var_names = ['F_15D_Cust_Pymt_Sum', 'F_15D_Cust_Pymt_Avg', 'F_15D_Cust_Pymt_Std', 'F_15D_Cust_Pymt_Cnt'])
        label = utils.utils.append_feature(label, payment_15d_add, fill_value = {'F_15D_Cust_Pymt_Sum' : -1, 'F_15D_Cust_Pymt_Avg' : -1, 'F_15D_Cust_Pymt_Std' : -1, 'F_15D_Cust_Pymt_Cnt' : 0}, dtypes = {'F_15D_Cust_Pymt_Cnt' : int})
    else:
        logger.info('\t 2.1. 출금거래 60일 Sum / Avg / Std / Count')
        label['F_60D_Cust_Pymt_Sum'] = float(-1)
        label['F_60D_Cust_Pymt_Avg'] = float(-1)
        label['F_60D_Cust_Pymt_Std'] = float(-1)
        label['F_60D_Cust_Pymt_Cnt'] = 0
        
        label['F_60D_Cust_Pymt_Sum'] = label['F_60D_Cust_Pymt_Sum'].astyp(float)
        label['F_60D_Cust_Pymt_Avg'] = label['F_60D_Cust_Pymt_Avg'].astyp(float)
        label['F_60D_Cust_Pymt_Std'] = label['F_60D_Cust_Pymt_Std'].astyp(float)
        label['F_60D_Cust_Pymt_Cnt'] = label['F_60D_Cust_Pymt_Cnt'].astype(int)
        
        logger.info('\t 2.2. 출금거래 30일 Sum / Avg / Std / Count')
        label['F_30D_Cust_Pymt_Sum'] = float(-1)
        label['F_30D_Cust_Pymt_Avg'] = float(-1)
        label['F_30D_Cust_Pymt_Std'] = float(-1)
        label['F_30D_Cust_Pymt_Cnt'] = 0
        
        label['F_30D_Cust_Pymt_Sum'] = label['F_30D_Cust_Pymt_Sum'].astype(float)
        label['F_30D_Cust_Pymt_Avg'] = label['F_30D_Cust_Pymt_Avg'].astype(float)
        label['F_30D_Cust_Pymt_Std'] = label['F_30D_Cust_Pymt_Std'].astype(float)
        label['F_30D_Cust_Pymt_Cnt'] = label['F_30D_Cust_Pymt_Cnt'].astype(int)
                
        logger.info('\t 2.3. 출금거래 15일 Sum / Avg / Std / Count')
        label['F_15D_Cust_Pymt_Sum'] = float(-1)
        label['F_15D_Cust_Pymt_Avg'] = float(-1)
        label['F_15D_Cust_Pymt_Std'] = float(-1)
        label['F_15D_Cust_Pymt_Cnt'] = 0
        
        label['F_15D_Cust_Pymt_Sum'] = label['F_15D_Cust_Pymt_Sum'].astyp(float)
        label['F_15D_Cust_Pymt_Avg'] = label['F_15D_Cust_Pymt_Avg'].astyp(float)
        label['F_15D_Cust_Pymt_Std'] = label['F_15D_Cust_Pymt_Std'].astyp(float)
        label['F_15D_Cust_Pymt_Cnt'] = label['F_15D_Cust_Pymt_Cnt'].astype(int)
    del payment_tr
    
    logger.info('\t 3. 입금거래 평균 ratio')
    label = utils.ratio_col(label, 'F_15D_Cust_Dep_Avg', 'F_30D_Cust_Dep_Avg', 'F_15D_div_by_30D_Dep_Avg_Ratio')
    label = utils.ratio_col(label, 'F_15D_Cust_Dep_Avg', 'F_60D_Cust_Dep_Avg', 'F_15D_div_by_60D_Dep_Avg_Ratio')
    label = utils.ratio_col(label, 'F_30D_Cust_Dep_Avg', 'F_60D_Cust_Dep_Avg', 'F_30D_div_by_60D_Dep_Avg_Ratio')
    
    logger.info('\t 4. 출금거래 평균 ratio')
    label = utils.ratio_col(label, 'F_15D_Cust_Pymt_Avg', 'F_30D_Cust_Pymt_Avg', 'F_15D_div_by_30D_Pymt_Avg_Ratio')
    label = utils.ratio_col(label, 'F_15D_Cust_Pymt_Avg', 'F_60D_Cust_Pymt_Avg', 'F_15D_div_by_60D_Pymt_Avg_Ratio')
    label = utils.ratio_col(label, 'F_30D_Cust_Pymt_Avg', 'F_60D_Cust_Pymt_Avg', 'F_30D_div_by_60D_Pymt_Avg_Ratio')
    
    logger.info('\t 5. 외환거래')
    fe_tr = data_fe.copy()    
    if 0 != fe_tr.shape[0]:
        fe_tr['cnt'] = 1
        fe_tr = utils.reindex_base_table(fe_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 5.1. 외환 거래 30일 Avg / Count')
        fe_30d_add = utils.add_rolling_feature(fe_tr, ['tram_us', 'cnt'], {'tram_us' : pkgs.np.mean, 'cnt' : sum}, 'tr_dt', '31D',
                                                     var_names = ['F_30D_Cust_Fe_Tr_Avg', 'F_30D_Cust_Fe_Tr_Cnt'])
        label = utils.append_feature(label, fe_30d_add, fill_value = {'F_30D_Cust_Fe_Tr_Avg' : -1, 'F_30D_Cust_Fe_Tr_Cnt' : 0}, dtypes = {'F_30D_Cust_Fe_Tr_Cnt' : int})
        
        logger.info('\t 5.2. 외환 거래 15일 Avg / Count')
        fe_15d_add = utils.add_rolling_feature(fe_tr, ['tram_us', 'cnt'], {'tram_us' : pkgs.np.mean, 'cnt' : sum}, 'tr_dt', '16D',
                                                     var_names = ['F_15D_Cust_Fe_Tr_Avg', 'F_15D_Cust_Fe_Tr_Cnt'])
        label = utils.append_feature(label, fe_15d_add, fill_value = {'F_15D_Cust_Fe_Tr_Avg' : -1, 'F_15D_Cust_Fe_Tr_Cnt' : 0}, dtypes = {'F_15D_Cust_Fe_Tr_Cnt' : int})
        
        logger.info('\t 5.3. 외환 거래 7일 Avg / Count')
        fe_7d_add = utils.add_rolling_feature(fe_tr, ['tram_us', 'cnt'], {'tram_us' : pkgs.np.mean, 'cnt' : sum}, 'tr_dt', '8D',
                                                    var_names = ['F_7D_Cust_Fe_Tr_Avg', 'F_7D_Cust_Fe_Tr_Cnt'])
        label = utils.append_feature(label, fe_7d_add, fill_value = {'F_7D_Cust_Fe_Tr_Avg' : -1, 'F_7D_Cust_Fe_Tr_Cnt' : 0}, dtypes = {'F_7D_Cust_Fe_Tr_Cnt' : int})
        
        logger.info('\t 5.3. 외환 거래 당일 Count')
        fe_1d_add = utils.add_rolling_feature(fe_tr, 'cnt', sum, 'tr_dt', '1D', var_names = 'F_1D_Cust_Fe_Tr_Cnt')
        label = utils.append_feature(label, fe_1d_add, fill_value = {'F_1D_Cust_Fe_Tr_Cnt' : 0}, dtypes = {'F_1D_Cust_Fe_Tr_Cnt' : int})
    else:
        logger.info('\t 5.1. 외환 거래 30일 Avg / Count')
        label['F_30D_Cust_Fe_Tr_Avg'] = float(-1)
        label['F_30D_Cust_Fe_Tr_Cnt'] = 0
        
        label['F_30D_Cust_Fe_Tr_Avg'] = label['F_30D_Cust_Fe_Tr_Avg'].astype(float)
        label['F_30D_Cust_Fe_Tr_Cnt'] = label['F_30D_Cust_Fe_Tr_Cnt'].astype(int)
        
        logger.info('\t 5.2. 외환 거래 15일 Avg / Count')
        label['F_15D_Cust_Fe_Tr_Avg'] = float(-1)
        label['F_15D_Cust_Fe_Tr_Cnt'] = 0
        
        label['F_15D_Cust_Fe_Tr_Avg'] = label['F_15D_Cust_Fe_Tr_Avg'].astype(float)
        label['F_15D_Cust_Fe_Tr_Cnt'] = label['F_15D_Cust_Fe_Tr_Cnt'].astype(int)
        
        logger.info('\t 5.1. 외환 거래 7일 Avg / Count')
        label['F_7D_Cust_Fe_Tr_Avg'] = float(-1)
        label['F_7D_Cust_Fe_Tr_Cnt'] = 0
        
        label['F_7D_Cust_Fe_Tr_Avg'] = label['F_7D_Cust_Fe_Tr_Avg'].astype(float)
        label['F_7D_Cust_Fe_Tr_Cnt'] = label['F_7D_Cust_Fe_Tr_Cnt'].astype(int)
        
        logger.info('\t 5.1. 외환 거래 당일 Count')
        label['F_1D_Cust_Fe_Tr_Cnt'] = 0
        label['F_1D_Cust_Fe_Tr_Cnt'] = label['F_1D_Cust_Fe_Tr_Cnt'].astype(int)
    del fe_tr
    
    logger.info('\t 5.5. 외환거래 평균 ratio')
    label = utils.ratio_col(label, 'F_7D_Cust_Fe_Tr_Avg', 'F_30D_Cust_Fe_Tr_Avg', 'F_7D_div_by_30D_Fe_Tr_Avg_Ratio')
    label = utils.ratio_col(label, 'F_7D_Cust_Fe_Tr_Avg', 'F_15D_Cust_Fe_Tr_Avg', 'F_7D_div_by_15D_Fe_Tr_Avg_Ratio')
    label = utils.ratio_col(label, 'F_15D_Cust_Fe_Tr_Avg', 'F_30D_Cust_Fe_Tr_Avg', 'F_15D_div_by_30D_Fe_Tr_Avg_Ratio')
    
    logger.info('\t 6. 비외환거래')
    nfe_tr = utils.filter_tr_data(data_nfe, [f"sbjc not in {cfg.NO_FE_CODE}"])
    nfe_tr = nfe_tr[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]    
    if 0 != nfe_tr.shape[0]:
        nfe_tr['cnt'] = 1
        nfe_tr = utils.reindex_base_table(nfe_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 6.1. 비외환거래 30일 Sum / Avg / Count')
        nfe_30d_add = utils.add_rolling_feature(nfe_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_cont = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '31D',
                                                var_names = ['F_30D_Cust_Nfe_Tr_Sum', 'F_30D_Cust_Nfe_Tr_Avg', 'F_30D_Cust_Nfe_Tr_Cnt'])
        label = utils.append_feature(label, nfe_30d_add, fill_value = {'F_30D_Cust_Nfe_Tr_Sum' : -1, 'F_30D_Cust_Nfe_Tr_Avg' : -1, 'F_30D_Cust_Nfe_Tr_Cnt' : 0}, dtypes = {'F_30D_Cust_Nfe_Tr_Cnt' : int})
        
        logger.info('\t 6.2. 비외환거래 7일 Sum / Avg / Count')
        nfe_7d_add = utils.add_rolling_feature(nfe_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_cont = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '8D',
                                               var_names = ['F_7D_Cust_Nfe_Tr_Sum', 'F_7D_Cust_Nfe_Tr_Avg', 'F_7D_Cust_Nfe_Tr_Cnt'])
        label = utils.append_feature(label, nfe_7d_add, fill_value = {'F_7D_Cust_Nfe_Tr_Sum' : -1, 'F_7D_Cust_Nfe_Tr_Avg' : -1, 'F_7D_Cust_Nfe_Tr_Cnt' : 0}, dtypes = {'F_7D_Cust_Nfe_Tr_Cnt' : int})
        
        logger.info('\t 6.3. 비외환거래 당일  Sum / Avg / Count')
        nfe_1d_add = utils.add_rolling_feature(nfe_tr, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_cont = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '1D',
                                               var_names = ['F_1D_Cust_Nfe_Tr_Sum', 'F_1D_Cust_Nfe_Tr_Avg', 'F_1D_Cust_Nfe_Tr_Cnt'])
        label = utils.append_feature(label, nfe_1d_add, fill_value = {'F_1D_Cust_Nfe_Tr_Sum' : -1, 'F_1D_Cust_Nfe_Tr_Avg' : -1, 'F_1D_Cust_Nfe_Tr_Cnt' : 0}, dtypes = {'F_1D_Cust_Nfe_Tr_Cnt' : int})
    else:
        logger.info('\t 6.1. 비외환거래 30일 Sum / Avg / Count')
        label['F_30D_Cust_Nfe_Tr_Sum'] = float(-1)
        label['F_30D_Cust_Nfe_Tr_Avg'] = float(-1)
        label['F_30D_Cust_Nfe_Tr_Cnt'] = 0
        
        label['F_30D_Cust_Nfe_Tr_Sum'] = label['F_30D_Cust_Nfe_Tr_Sum'].astype(float)
        label['F_30D_Cust_Nfe_Tr_Avg'] = label['F_30D_Cust_Nfe_Tr_Avg'].astype(float)
        label['F_30D_Cust_Nfe_Tr_Cnt'] = label['F_30D_Cust_Nfe_Tr_Cnt'].astype(int)
        
        logger.info('\t 6.2. 비외환거래 7일 Sum / Avg / Count')
        label['F_7D_Cust_Nfe_Tr_Sum'] = float(-1)
        label['F_7D_Cust_Nfe_Tr_Avg'] = float(-1)
        label['F_7D_Cust_Nfe_Tr_Cnt'] = 0
        
        label['F_7D_Cust_Nfe_Tr_Sum'] = label['F_7D_Cust_Nfe_Tr_Sum'].astype(float)
        label['F_7D_Cust_Nfe_Tr_Avg'] = label['F_7D_Cust_Nfe_Tr_Avg'].astype(float)
        label['F_7D_Cust_Nfe_Tr_Cnt'] = label['F_7D_Cust_Nfe_Tr_Cnt'].astype(int)
        
        logger.info('\t 6.3. 비외환거래 당일 Sum / Avg / Count')
        label['F_1D_Cust_Nfe_Tr_Sum'] = float(-1)
        label['F_1D_Cust_Nfe_Tr_Avg'] = float(-1)
        label['F_1D_Cust_Nfe_Tr_Cnt'] = 0
        
        label['F_1D_Cust_Nfe_Tr_Sum'] = label['F_1D_Cust_Nfe_Tr_Sum'].astype(float)
        label['F_1D_Cust_Nfe_Tr_Avg'] = label['F_1D_Cust_Nfe_Tr_Avg'].astype(float)
        label['F_1D_Cust_Nfe_Tr_Cnt'] = label['F_1D_Cust_Nfe_Tr_Cnt'].astype(int)
    del nfe_tr
    
    logger.info('\t 7. 비대면채널 이용 거래')
    nftf_tr = utils.filter_tr_data(data_fe, [f"tr_metric in {cfg.NFTF_CODE}"])
    nftf_tr = nftf_tr[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    if 0 != nftf_tr.shape[0]:
        nftf_tr['cnt'] = 1
        nftf_tr = utils.reindex_base_table(nftf_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 7.1. 비대면채널 이용 거래 7일 Count')
        nftf_7d_add = utils.add_rolling_feature(nftf_tr, 'cnt', sum, 'tr_dt', '8D', var_names = 'F_7D_Cust_Nftf_Tr_Cnt')
        label = utils.append_feature(label, nftf_7d_add, fill_value = {'F_7D_Cust_Nftf_Tr_Cnt' : 0}, dtypes = {'F_7D_Cust_Nftf_Tr_Cnt' : int})
        
        logger.info('\t 7.2. 비대면채널 이용 거래 당일 Count')
        nftf_1d_add = utils.add_rolling_feature(nftf_tr, 'cnt', sum, 'tr_dt', '1D', var_names = 'F_1D_Cust_Nftf_Tr_Cnt')
        label = utils.append_feature(label, nftf_1d_add, fill_value = {'F_1D_Cust_Nftf_Tr_Cnt' : 0}, dtypes = {'F_1D_Cust_Nftf_Tr_Cnt' : int})
    else:
        logger.info('\t 7.1. 비대면채널 이용 거래 7일 Count')
        label['F_7D_Cust_Nftf_Tr_Cnt'] = 0
        label['F_7D_Cust_Nftf_Tr_Cnt'] = label['F_7D_Cust_Nftf_Tr_Cnt'].astype(int)
    
        logger.info('\t 7.1. 비대면채널 이용 거래 당일 Count')
        label['F_1D_Cust_Nftf_Tr_Cnt'] = 0
        label['F_1D_Cust_Nftf_Tr_Cnt'] = label['F_1D_Cust_Nftf_Tr_Cnt'].astype(int)
    
    logger.info('\t 7.5. 외환 전체 거래 대비 비대면채널 이용 거래 7일 Ratio')
    label = utils.ratio_col(label, 'F_7D_Cust_Nftf_Tr_Cnt', 'F_7D_Cust_Fe_Tr_Cnt', 'F_7D_Cust_Nftf_Tr_Cnt_Ratio')
    label = label.drop(columns = ['F_7D_Cust_Nftf_Tr_Cnt'], axis = 1)
    
    logger.info('\t 7.5. 외환 전체 거래 대비 비대면채널 이용 거래 당일 Ratio')
    label = utils.ratio_col(label, 'F_1D_Cust_Nftf_Tr_Cnt', 'F_1D_Cust_Fe_Tr_Cnt', 'F_1D_Cust_Nftf_Tr_Cnt_Ratio')
    label = label.drop(columns = ['F_1D_Cust_Nftf_Tr_Cnt'], axis = 1)
    
    del nftf_tr
    
    logger.info('\t 8. 환전거래')
    exch_tr = utils.filter_tr_data(data, [f"cptld_tr_kdc in {cfg.EXCH_CODE}"])
    exch_tr = exch_tr[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    if 0 != exch_tr.shape[0]:
        exch_tr['cnt'] = 1
        exch_tr = utils.reindex_base_table(exch_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 8.1. 환전거래 30일 Sum / Count')
        exch_30d_add = utils.add_rolling_feature(exch_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt' : sum}, 'tr_dt', '31D', var_names = ['F_30D_Cust_Exch_Tr_Sum', 'F_30D_Cust_Exch_Tr_Cnt'])
        label = utils.append_feature(label, exch_30d_add, fill_value = {'F_30D_Cust_Exch_Tr_Sum' : -1, 'F_30D_Cust_Exch_Tr_Cnt' : 0}, dtypes = {'F_1D_Cust_Exch_Tr_Cnt' : int})
        
        logger.info('\t 8.3. 환전거래 7일 Sum / Count')
        exch_7d_add = utils.add_rolling_feature(exch_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt' : sum}, 'tr_dt', '8D', var_names = ['F_7D_Cust_Exch_Tr_Sum', 'F_1D_Cust_Exch_Tr_Cnt'])
        label = utils.append_feature(label, exch_7d_add, fill_value = {'F_7D_Cust_Exch_Tr_Sum' : -1, 'F_7D_Cust_Exch_Tr_Cnt' : 0}, dtypes = {'F_7D_Cust_Exch_Tr_Cnt' : int})
        
        logger.info('\t 8.3. 환전거래 당일 Sum / Count')
        exch_1d_add = utils.add_rolling_feature(exch_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt' : sum}, 'tr_dt', '1D', var_names = ['F_1D_Cust_Exch_Tr_Sum', 'F_1D_Cust_Exch_Tr_Cnt'])
        label = utils.append_feature(label, exch_1d_add, fill_value = {'F_1D_Cust_Exch_Tr_Sum' : -1, 'F_1D_Cust_Exch_Tr_Cnt' : 0}, dtypes = {'F_1D_Cust_Exch_Tr_Cnt' : int})
    else:
        logger.info('\t 8.1. 환전거래 30일 Sum / Count')
        label['F_30D_Cust_Exch_Tr_Sum'] = float(-1)
        label['F_30D_Cust_Exch_Tr_Cnt'] = 0
        
        label['F_30D_Cust_Exch_Tr_Sum'] = label['F_30D_Cust_Exch_Tr_Sum'].astype(float)
        label['F_30D_Cust_Exch_Tr_Cnt'] = label['F_30D_Cust_Exch_Tr_Cnt'].astype(int)
        
        logger.info('\t 8.2. 환전거래 7일 Sum / Count')
        label['F_7D_Cust_Exch_Tr_Sum'] = float(-1)
        label['F_7D_Cust_Exch_Tr_Cnt'] = 0
        
        label['F_7D_Cust_Exch_Tr_Sum'] = label['F_7D_Cust_Exch_Tr_Sum'].astype(float)
        label['F_7D_Cust_Exch_Tr_Cnt'] = label['F_7D_Cust_Exch_Tr_Cnt'].astype(int)
        
        logger.info('\t 8.3. 환전거래 당일 Sum / Count')
        label['F_1D_Cust_Exch_Tr_Sum'] = float(-1)
        label['F_1D_Cust_Exch_Tr_Cnt'] = 0
        
        label['F_1D_Cust_Exch_Tr_Sum'] = label['F_1D_Cust_Exch_Tr_Sum'].astype(float)
        label['F_1D_Cust_Exch_Tr_Cnt'] = label['F_1D_Cust_Exch_Tr_Cnt'].astype(int)
    del exch_tr
    
    logger.info('\t 9. 거래계좌 정보')
    acno_tr = data_fe.copy()
    if 0 != acno_tr.shape[0]:
        acno_tr['cnt'] = 1
        acno_tr = utils.handling_acno_type_with_none(acno_tr, ['acno'], {'acno' : 'new_acno'})
        acno_tr = utils.reindex_base_table(acno_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        data_collect = []
        
        logger.info('\t 9.1. 거래계좌 30일 Count')
        acno_30d_add = acno_tr.set_index('tr_dt').sort_index().groupby('cusno')['new_acno'].rolling('31D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_indeX()
        acno_30d_add = acno_30d_add.rename(columns = {'new_acno' : 'F_30D_Cust_Acno_Cnt'})
        data_collect.append(acno_30d_add)
        
        logger.info('\t 9.2. 거래계좌 7일 Count')
        acno_7d_add = acno_tr.set_index('tr_dt').sort_index().groupby('cusno')['new_acno'].rolling('8D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_indeX()
        acno_7d_add = acno_7d_add.rename(columns = {'new_acno' : 'F_7D_Cust_Acno_Cnt'})
        data_collect.append(acno_7d_add)
        
        logger.info('\t 9.3. 거래계좌 3일 Count')
        acno_3d_add = acno_tr.set_index('tr_dt').sort_index().groupby('cusno')['new_acno'].rolling('4D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_indeX()
        acno_3d_add = acno_3d_add.rename(columns = {'new_acno' : 'F_3D_Cust_Acno_Cnt'})
        data_collect.append(acno_3d_add)
        
        logger.info('\t 9.4. 거래계좌 당일 Count')
        acno_1d_add = acno_tr.set_index('tr_dt').sort_index().groupby(['cusno', 'tr_dt'])['new_acno'].apply(value_select).reset_index()
        acno_1d_add = acno_1d_add.rename(columns = {'new_acno' : 'F_1D_Cust_Acno_Cnt'})
        data_collect.append(acno_1d_add)
        
        columns_name = ['F_30D_Cust_Acno_Cnt', 'F_7D_Cust_Acno_Cnt', 'F_3D_Cust_Acno_Cnt', 'F_1D_Cust_Acno_Cnt']
        for idx, val in enumerate(data_collect):
            label = utils.append(label, val, fill_value = {columns_name[idx] : 0}, dtypes = {columns_name[idx] : int})
    else:
        logger.info('\t 9.1. 거래계좌 30일 Count')
        label['F_30D_Cust_Acno_Cnt'] = 0
        label['F_30D_Cust_Acno_Cnt'] = label['F_30D_Cust_Acno_Cnt'].astype(int)
        
        logger.info('\t 9.2. 거래계좌 7일 Count')
        label['F_7D_Cust_Acno_Cnt'] = 0
        label['F_7D_Cust_Acno_Cnt'] = label['F_7D_Cust_Acno_Cnt'].astype(int)
        
        logger.info('\t 9.3. 거래계좌 3일 Count')
        label['F_3D_Cust_Acno_Cnt'] = 0
        label['F_3D_Cust_Acno_Cnt'] = label['F_3D_Cust_Acno_Cnt'].astype(int)
        
        logger.info('\t 9.4. 거래계좌 당일 Count')
        label['F_1D_Cust_Acno_Cnt'] = 0
        label['F_1D_Cust_Acno_Cnt'] = label['F_1D_Cust_Acno_Cnt'].astype(int)
    del acno_tr
    
    logger.info('\t 10. 출금거래 - withdraw')
    withdraw_tr = utils.filter_tr_data(data_nfe, [f"cptld_tr_kdc in {cfg.WITHDRAW_CODE}"])
    withdraw_tr = withdraw_tr[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    if 0 != withdraw_tr.shape[0]:
        withdraw_tr['cnt'] = 1
        withdraw_tr = utils.reindex_base_table(withdraw_tr, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 10.1. 출금 거래 30일 Sum / Count')
        withdraw_30d_add = utils.add_rolling_feature(withdraw_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt'  : sum}, 'tr_dt', '31D',
                                                     var_names = ['F_30D_Cust_Wd_Sum', 'F_30D_Cust_Wd_Cnt'])
        label = utils.append_feature(label, withdraw_30d_add, fill_value = {'F_30D_Cust_Wd_Sum' : -1, 'F_30D_Cust_Wd_Cnt' : 0}, dtypes = {'F_30D_Cust_Wd_Cnt' : int})
        
        logger.info('\t 10.2. 출금 거래 7일 Sum / Count')
        withdraw_7d_add = utils.add_rolling_feature(withdraw_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt'  : sum}, 'tr_dt', '8D',
                                              var_names = ['F_7D_Cust_Wd_Sum', 'F_7D_Cust_Wd_Cnt'])
        label = utils.append_feature(label, withdraw_7d_add, fill_value = {'F_7D_Cust_Wd_Sum' : -1, 'F_7D_Cust_Wd_Cnt' : 0}, dtypes = {'F_7D_Cust_Wd_Cnt' : int})
        
        logger.info('\t 10.3. 출금 거래 3일 Sum / Count')
        withdraw_3d_add = utils.add_rolling_feature(withdraw_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt'  : sum}, 'tr_dt', '4D',
                                                    var_names = ['F_3D_Cust_Wd_Sum', 'F_3D_Cust_Wd_Cnt'])
        label = utils.append_feature(label, withdraw_3d_add, fill_value = {'F_3D_Cust_Wd_Sum' : -1, 'F_3D_Cust_Wd_Cnt' : 0}, dtypes = {'F_3D_Cust_Wd_Cnt' : int})
        
        logger.info('\t 10.4. 출금 거래 당일 Sum / Count')
        withdraw_1d_add = utils.add_rolling_feature(withdraw_tr, ['tram_us', 'cnt'], {'tram_us' : lambda x: x.sum(min_count = 1), 'cnt'  : sum}, 'tr_dt', '1D',
                                                    var_names = ['F_1D_Cust_Wd_Sum', 'F_1D_Cust_Wd_Cnt'])
        label = utils.append_feature(label, withdraw_1d_add, fill_value = {'F_1D_Cust_Wd_Sum' : -1, 'F_1D_Cust_Wd_Cnt' : 0}, dtypes = {'F_1D_Cust_Wd_Cnt' : int})
    else:
        logger.info('\t 10.1. 출금 거래 30일 Sum / Count')
        label['F_30D_Cust_Wd_Sum'] = float(-1)
        label['F_30D_Cust_Wd_Cnt'] = 0
        
        label['F_30D_Cust_Wd_Sum'] = label['F_30D_Cust_Wd_Sum'].astype(float)
        label['F_30D_Cust_Wd_Cnt'] = label['F_30D_Cust_Wd_Cnt'].astype(int)
        
        logger.info('\t 10.2. 출금 거래 7일 Sum / Count')
        label['F_7D_Cust_Wd_Sum'] = float(-1)
        label['F_7D_Cust_Wd_Cnt'] = 0
        
        label['F_7D_Cust_Wd_Sum'] = label['F_7D_Cust_Wd_Sum'].astype(float)
        label['F_7D_Cust_Wd_Cnt'] = label['F_7D_Cust_Wd_Cnt'].astype(int)
        
        logger.info('\t 10.3. 출금 거래 3일 Sum / Count')
        label['F_3D_Cust_Wd_Sum'] = float(-1)
        label['F_3D_Cust_Wd_Cnt'] = 0
        
        label['F_3D_Cust_Wd_Sum'] = label['F_3D_Cust_Wd_Sum'].astype(float)
        label['F_3D_Cust_Wd_Cnt'] = label['F_3D_Cust_Wd_Cnt'].astype(int)
        
        logger.info('\t 10.4. 출금 거래 당일 Sum / Count')
        label['F_1D_Cust_Wd_Sum'] = float(-1)
        label['F_1D_Cust_Wd_Cnt'] = 0
        
        label['F_1D_Cust_Wd_Sum'] = label['F_1D_Cust_Wd_Sum'].astype(float)
        label['F_1D_Cust_Wd_Cnt'] = label['F_1D_Cust_Wd_Cnt'].astype(int)
    del withdraw_tr
    
    logger.info('\t Part. 2')
    
    logger.info('\t 1. 최근 60일 동안 혐의 이력이 있는 고객들의 거래계좌수, 송금거래 Sum / Count, 출금거래 Sum / Count')
    logger.info('\t 1.1. RZT (STR-혐의거래추출결과) & DCZ (공통-결재) Merge')
    merged_rzt_dcz_data = utils.merged_rzt_dcz(rzt_data, dcz_data, target_cust, label = 'sus')
    
    logger.info('\t 1.2. 최근 60일 동안 고객 별 혐의이력 Count')
    rzt_60d_cnt = utils.add_rolling_feature(merged_rzt_dcz_data, 'cnt', sum, 'tr_dt', '61D', var_names = '60D_Cust_Rzt_Cnt', dtypes = int)
    tmp_rzt_60d_cnt = rzt_60d_cnt[rzt_60d_cnt['60D_Cust_Rzt_Cnt'] > 0][['cusno', 'tr_dt']].drop_duplicates()
    tmp_rzt_60d_cnt['label'] = 1
    
    unique_rzt = merged_rzt_dcz_data[merged_rzt_dcz_data['cnt'] > 0].drop_duplicates(['cusno', 'dcz_rqr_dt'], keep = 'last').sort_values('dcz_rqr_dt').drop(['tr_dt', 'dcz_sqno', 'intg_imps_key_val'], axis = 1)
    
    rzt_label_date = pd.merge_asof(label[['cusno', 'tr_dt']].sort_values('tr_dt'), unique_rzt, left_on = 'tr_dt', right_on = 'dcz_rqr_dt', by = 'cusno').dropna(subset = ['sspn_tr_stsc'])
    rzt_label_date['rzt_period'] = (rzt_label_date['tr_dt'] - rzt_label_date['dcz_rqr_dt']).dt.days
    
    rzt_period = rzt_label_date[['cusno', 'tr_dt', 'rzt_period']].drop_duplicates()
    rzt_period = rzt_period.reset_index(drop = True)
    
    rzt_label_date = rzt_label_date[rzt_label_date['rzt_period'] <= 60]
    
    logger.info('\t 1.3. 검출일 기준 과거 60일 동안 혐의거래가 있는 고객 중')
    if 0 != rzt_label_date.shape[0]:
        logger.info('\t 1.3.1. 외환거래에 사용한 수신 계좌 수')
        # 임출금거래에 사용하는 DB
        acno_tr_data_in = utils.sus_60d_acno_tr(data_nfe, label)
        rzt_acno_30d_tr_in = utils.matching_trlog_tr_date(rzt_label_date, acno_tr_data_in, ['acno', 'acno_int'], 33)        
        rzt_acno_30d_tr_in.loc[rzt_acno_30d_tr_in['acno'].isna(), 'cnt'] = 0
        rzt_acno_30d_tr_in.loc[rzt_acno_30d_tr_in['cnt'].isna(), 'cnt'] = 1        
        rzt_acno_30d_tr_in['cnt'] = rzt_acno_30d_tr_in['cnt'].astype(int)
        
        data_collect = [] 
        
        rzt_acno_30d_add = rzt_acno_30d_tr_in.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('31D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_30d_add = rzt_acno_30d_add.rename(columns = {'acno_int' : 'F_30D_Cust_Nfe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_30d_add)
        
        rzt_acno_7d_add = rzt_acno_30d_tr_in.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('8D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_7d_add = rzt_acno_7d_add.rename(columns = {'acno_int' : 'F_7D_Cust_Nfe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_7d_add)
        
        rzt_acno_3d_add = rzt_acno_30d_tr_in.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('4D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_3d_add = acno_3d_add.rename(columns = {'acno_int' : 'F_3D_Cust_Nfe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_3d_add)
                
        rzt_acno_1d_add = rzt_acno_30d_tr_in.set_index('tr_dt').sort_index().groupby(['cusno', 'tr_dt'])['acno_int'].apply(value_select).reset_index()
        rzt_acno_1d_add = acno_1d_add.rename(columns = {'acno_int' : 'F_1D_Cust_Nfe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_1d_add)
        
        pre_label = rzt_label_date.copy()[['cusno', 'tr_dt']]
        for data in data_collect:
            pre_label = pre_label.merge(data, on = ['cusno', 'tr_dt'], how = 'left')
        label = utils.append(label, pre_label,
                             fill_value = {'F_1D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : 0, 'F_3D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : 0, 'F_7D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : 0, 'F_30D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : 0},
                             dtypes = {'F_1D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : int, 'F_3D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : int, 'F_7D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : int, 'F_30D_Cust_Nfe_Acno_Cnt_by_60D_Sus' : int})
        del acno_tr_data_in
        del rzt_acno_30d_tr_in
        del rzt_acno_30d_add
        del rzt_acno_7d_add
        del rzt_acno_3d_add
        del rzt_acno_1d_add
        del data_collect
        del pre_label
        
        logger.info('\t 1.3.2. 외환거래에서 사용한 외환 계좌 수')
        acno_tr_data_fe = utils.sus_60d_acno_tr(data_fe, label)
        rzt_acno_30d_tr_fe = utils.matching_trlog_tr_date(rzt_label_date, acno_tr_data_fe, ['acno', 'acno_int'], 33)        
        rzt_acno_30d_tr_fe.loc[rzt_acno_30d_tr_fe['acno'].isna(), 'cnt'] = 0
        rzt_acno_30d_tr_fe.loc[rzt_acno_30d_tr_fe['cnt'].isna(), 'cnt'] = 1        
        rzt_acno_30d_tr_fe['cnt'] = rzt_acno_30d_tr_fe['cnt'].astype(int)
        
        data_collect = []
        
        rzt_acno_30d_add = rzt_acno_30d_tr_fe.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('31D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_30d_add = rzt_acno_30d_add.rename(columns = {'acno_int' : 'F_30D_Cust_Fe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_30d_add)
        
        rzt_acno_7d_add = rzt_acno_30d_tr_fe.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('8D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_7d_add = rzt_acno_7d_add.rename(columns = {'acno_int' : 'F_7D_Cust_Fe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_7d_add)
        
        rzt_acno_3d_add = rzt_acno_30d_tr_fe.set_index('tr_dt').sort_index().groupby('cusno')['acno_int'].rolling('4D', min_periods = 1).apply(value_select).reset_index().groupby(['cusno', 'tr_dt']).max().reset_index()
        rzt_acno_3d_add = rzt_acno_3d_add.rename(columns = {'acno_int' : 'F_3D_Cust_Fe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_3d_add)
        
        rzt_acno_1d_add = rzt_acno_30d_tr_fe.set_index('tr_dt').sort_index().groupby(['cusno', 'tr_dt'])['acno_int'].apply(value_select).reset_index()
        rzt_acno_1d_add = rzt_acno_1d_add.rename(columns = {'acno_int' : 'F_1D_Cust_Fe_Acno_Cnt_by_60D_Sus'})
        data_collect.append(rzt_acno_1d_add)
        
        pre_label = rzt_label_date.copy()[['cusno', 'tr_dt']]
        for data in data_collect:
            pre_label = pre_label.merge(data, on = ['cusno', 'tr_dt'], how = 'left')
        label = utils.append_feature(label, pre_label,
                                     fill_value = {'F_1D_Cust_Fe_Acno_Cnt_by_60D_Sus' : 0, 'F_3D_Cust_Fe_Acno_Cnt_by_60D_Sus' : 0, 'F_7D_Cust_Fe_Acno_Cnt_by_60D_Sus' : 0, 'F_30D_Cust_Fe_Acno_Cnt_by_60D_Sus' : 0},
                                     dtypes = {'F_1D_Cust_Fe_Acno_Cnt_by_60D_Sus' : int, 'F_3D_Cust_Fe_Acno_Cnt_by_60D_Sus' : int, 'F_7D_Cust_Fe_Acno_Cnt_by_60D_Sus' : int, 'F_30D_Cust_Fe_Acno_Cnt_by_60D_Sus' : int})
        del acno_tr_data_fe
        del rzt_acno_30d_tr_fe
        del rzt_acno_30d_add
        del rzt_acno_7d_add
        del rzt_acno_3d_add
        del rzt_acno_1d_add
        del data_collect
        del pre_label
        
        logger.info('\t 1.3.3. 당발거래 Sum / Count')
        data_outward = data_fe[data_fe.sbjc == '486']
        data_outward['cnt'] = 1
        data_outward = utils.reindex_base_table(data_outward, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_types = int)        
        rzt_cus_30d_outward = utils.matching_trlog_tr_date(rzt_label_date, data_outward, ['tram_us', 'usdc_tram', 'cnt'], 33)
        
        data_collect = []
        
        rzt_outward_30d_add = utils.add_rolling_feature(rzt_cus_30d_outward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '31D',
                                                      var_names = ['F_30D_Cust_Outward_Sum_by_60D_Sus', 'F_30D_Cust_Outward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_outward_30d_add)
        
        rzt_outward_7d_add = utils.add_rolling_feature(rzt_cus_30d_outward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '8D',
                                                     var_names = ['F_7D_Cust_Outward_Sum_by_60D_Sus', 'F_7D_Cust_Outward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_outward_7d_add)
        
        rzt_outward_3d_add = utils.add_rolling_feature(rzt_cus_30d_outward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '4D',
                                                     var_names = ['F_3D_Cust_Outward_Sum_by_60D_Sus', 'F_3D_Cust_Outward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_outward_3d_add)
        
        rzt_outward_1d_add = utils.add_rolling_feature(rzt_cus_30d_outward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                                     var_names = ['F_1D_Cust_Outward_Sum_by_60D_Sus', 'F_1D_Cust_Outward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_outward_1d_add)
        
        pre_label = rzt_label_date.copy()[['cusno', 'tr_dt']]
        for data in data_collect:
            pre_label = pre_label.merge(data, on = ['cusno', 'tr_dt'], how = 'left')
        label = utils.append_feature(label, pre_label,
                                     fill_value = {'F_1D_Cust_Outward_Sum_by_60D_Sus' : 0, 'F_3D_Cust_Outward_Sum_by_60D_Sus' : 0, 'F_7D_Cust_Outward_Sum_by_60D_Sus' : 0, 'F_30D_Cust_Outward_Sum_by_60D_Sus' : 0,
                                             'F_1D_Cust_Outward_Cnt_by_60D_Sus' : -1, 'F_3D_Cust_Outward_Cnt_by_60D_Sus' : -1, 'F_7D_Cust_Outward_Cnt_by_60D_Sus' : -1, 'F_30D_Cust_Outward_Cnt_by_60D_Sus' : -1},
                                     dtypes = {'F_1D_Cust_Outward_Cnt_by_60D_Sus' : int, 'F_3D_Cust_Outward_Cnt_by_60D_Sus' : int, 'F_7D_Cust_Outward_Cnt_by_60D_Sus' : int, 'F_30D_Cust_Outward_Cnt_by_60D_Sus' : int})
        del data_outward
        del rzt_cus_30d_outward
        del rzt_outward_30d_add
        del rzt_outward_7d_add
        del rzt_outward_3d_add
        del rzt_outward_1d_add
        del data_collect
        del pre_label
        
        logger.info('\t 1.3.4. 타발거래 Sum / Count')
        data_inward = data_fe[data_fe.sbjc == '487']
        data_inward['cnt'] = 1
        data_inward = utils.reindex_base_table(data_inward, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        rzt_cus_30d_inward = matching_trlog_tr_date(rzt_label_date, data_inward, ['tram_us', 'usdc_tram', 'cnt'], 33)
        
        data_collect = []
        
        rzt_inward_30d_add = utils.add_rolling_feature(rzt_cus_30d_inward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '31D',
                                                 var_names = ['F_30D_Cust_Inward_Sum_by_60D_Sus', 'F_30D_Cust_InOutward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_inward_30d_add)
        
        rzt_inward_7d_add = utils.add_rolling_feature(rzt_cus_30d_inward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '8D',
                                                var_names = ['F_7D_Cust_Inward_Sum_by_60D_Sus', 'F_7D_Cust_InOutward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_inward_7d_add)
        
        rzt_inward_3d_add = utils.add_rolling_feature(rzt_cus_30d_inward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '4D',
                                                var_names = ['F_3D_Cust_Inward_Sum_by_60D_Sus', 'F_3D_Cust_InOutward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_inward_3d_add)
        
        rzt_inward_1d_add = utils.add_rolling_feature(rzt_cus_30d_inward, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                                var_names = ['F_1D_Cust_Inward_Sum_by_60D_Sus', 'F_1D_Cust_InOutward_Cnt_by_60D_Sus'])
        data_collect.append(rzt_inward_1d_add)
        
        pre_label = rzt_label_date.copy()[['cusno', 'tr_dt']]
        for data in data_collect:
            pre_label = pre_label.merge(data, on = ['cusno', 'tr_dt'], how = 'left')
        label = utils.append_feature(label, pre_label,
                                     fill_value = {'F_1D_Cust_Inward_Sum_by_60D_Sus' : 0, 'F_3D_Cust_Inward_Sum_by_60D_Sus' : 0, 'F_7D_Cust_Inward_Sum_by_60D_Sus' : 0, 'F_30D_Cust_Inward_Sum_by_60D_Sus' : 0,
                                             'F_1D_Cust_Inward_Cnt_by_60D_Sus' : -1, 'F_3D_Cust_Inward_Cnt_by_60D_Sus' : -1, 'F_7D_Cust_Inward_Cnt_by_60D_Sus' : -1, 'F_30D_Cust_Inward_Cnt_by_60D_Sus' : -1},
                                     dtypes = {'F_1D_Cust_Inward_Cnt_by_60D_Sus' : int, 'F_3D_Cust_Inward_Cnt_by_60D_Sus' : int, 'F_7D_Cust_Inward_Cnt_by_60D_Sus' : int, 'F_30D_Cust_Inward_Cnt_by_60D_Sus' : int})
        
        del data_inward
        del rzt_cus_30d_inward
        del rzt_inward_30d_add
        del rzt_inward_7d_add
        del rzt_inward_3d_add
        del rzt_inward_1d_add
        
        logger.info('\t 1.3.5. 출금거래')
        withd_tr_data = utils.sus_60d_withdraw_tr(data_nfe, label)
        rzt_cus_30d_wd = utils.matching_trlog_tr_date(rzt_label_date, withd_tr_data, ['tram_us', 'usdc_tram', 'cnt'], 33)
        
        data_collect = []
                
        rzt_wd_30d_add = utils.add_rolling_feature(rzt_cus_30d_wd, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '31D',
                                                   var_names = ['F_30D_Cust_Wd_Sum_by_60D_Sus', 'F_31D_Cust_Wd_Cnt_by_60D_Sus'])
        data_collect.append(rzt_wd_30d_add)
        
        rzt_wd_7d_add = add_rolling_feature(rzt_cus_30d_wd, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '8D',
                                            var_names = ['F_7D_Cust_Wd_Sum_by_60D_Sus', 'F_7D_Cust_Wd_Cnt_by_60D_Sus'])
        data_collect.append(rzt_wd_7d_add)
        
        rzt_wd_3d_add = utils.add_rolling_feature(rzt_cus_30d_wd, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '4D',
                                              var_names = ['F_3D_Cust_Wd_Sum_by_60D_Sus', 'F_3D_Cust_Wd_Cnt_by_60D_Sus'])
        data_collect.append(rzt_wd_3d_add)
        
        rzt_wd_1d_add = utils.add_rolling_feature(rzt_cus_30d_wd, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                              var_names = ['F_1D_Cust_Wd_Sum_by_60D_Sus', 'F_1D_Cust_Wd_Cnt_by_60D_Sus'])
        data_collect.append(rzt_wd_1d_add)
        
        pre_label = rzt_label_date.copy()[['cusno', 'tr_dt']]
        for data in data_collect:
            pre_label = pre_label.merge(data, on = ['cusno', 'tr_dt'], how = 'left')
        label = utils.append_feature(label, pre_label,
                                     fill_value = {'F_1D_Cust_Wd_Sum_by_60D_Sus' : 0, 'F_3D_Cust_Wd_Sum_by_60D_Sus' : 0, 'F_7D_Cust_Wd_Sum_by_60D_Sus' : 0, 'F_30D_Cust_Wd_Sum_by_60D_Sus' : 0,
                                                   'F_1D_Cust_Wd_Cnt_by_60D_Sus' : -1, 'F_3D_Cust_Wd_Cnt_by_60D_Sus' : -1, 'F_7D_Cust_Wd_Cnt_by_60D_Sus' : -1, 'F_30D_Cust_Wd_Cnt_by_60D_Sus' : -1},
                                     dtypes = {'F_1D_Cust_Wd_Cnt_by_60D_Sus' : int, 'F_3D_Cust_Wd_Cnt_by_60D_Sus' : int, 'F_7D_Cust_Wd_Cnt_by_60D_Sus' : int, 'F_30D_Cust_Wd_Cnt_by_60D_Sus' : int})
        del withd_tr_data
        del rzt_cus_30d_wd
        del rzt_wd_30d_add
        del rzt_wd_7d_add
        del rzt_wd_3d_add
        del rzt_wd_1d_add
        del data_collect
        del pred_label
    else:
        logger.info('\t 1.3.1. 외환거래에 사용한 수신 계좌 수')
        label['F_1D_Cust_Nfe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_3D_Cust_Nfe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_7D_Cust_Nfe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_30D_Cust_Nfe_Acno_Cnt_by_60D_Sus'] = 0        
        
        logger.info('\t 1.3.2. 외환거래에서 사용한 외환 계좌 수')
        label['F_1D_Cust_Fe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_3D_Cust_Fe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_7D_Cust_Fe_Acno_Cnt_by_60D_Sus'] = 0
        label['F_30D_Cust_Fe_Acno_Cnt_by_60D_Sus'] = 0
        
        logger.info('\t 1.3.3. 당발거래 Sum / Count')
        label['F_1D_Cust_Outward_Sum_by_60_Sus'] = float(-1)
        label['F_3D_Cust_Outward_Sum_by_60_Sus'] = float(-1)
        label['F_7D_Cust_Outward_Sum_by_60_Sus'] = float(-1)
        label['F_30D_Cust_Outward_Sum_by_60_Sus'] = float(-1)
        
        label['F_1D_Cust_Outward_Cnt_by_60_Sus'] = 0
        label['F_3D_Cust_Outward_Cnt_by_60_Sus'] = 0
        label['F_7D_Cust_Outward_Cnt_by_60_Sus'] = 0
        label['F_30D_Cust_Outward_Cnt_by_60_Sus'] = 0
        
        logger.info('\t 1.3.4. 타발거래 Sum / Count')
        label['F_1D_Cust_Inward_Sum_by_60_Sus'] = float(-1)
        label['F_3D_Cust_Inward_Sum_by_60_Sus'] = float(-1)
        label['F_7D_Cust_Inward_Sum_by_60_Sus'] = float(-1)
        label['F_30D_Cust_Inward_Sum_by_60_Sus'] = float(-1)
        
        label['F_1D_Cust_Inward_Cnt_by_60_Sus'] = 0
        label['F_3D_Cust_Inward_Cnt_by_60_Sus'] = 0
        label['F_7D_Cust_Inward_Cnt_by_60_Sus'] = 0
        label['F_30D_Cust_Inward_Cnt_by_60_Sus'] = 0
        
        logger.info('\t 1.3.5. 출금 거래')
        label['F_1D_Cust_Wd_Sum_by_60_Sus'] = float(-1)
        label['F_3D_Cust_Wd_Sum_by_60_Sus'] = float(-1)
        label['F_7D_Cust_Wd_Sum_by_60_Sus'] = float(-1)
        label['F_30D_Cust_Wd_Sum_by_60_Sus'] = float(-1)
        
        label['F_1D_Cust_Wd_Cnt_by_60_Sus'] = 0
        label['F_3D_Cust_Wd_Cnt_by_60_Sus'] = 0
        label['F_7D_Cust_Wd_Cnt_by_60_Sus'] = 0
        label['F_30D_Cust_Wd_Cnt_by_60_Sus'] = 0
    
    loger.info('\t 1.4. STR 보고 관련 변수')
    sus_rzt_dcz_data = utils.merge_rzt_dcz(rzt_data, dcz_data, target_cust, label, 'sus')
    nosus_rzt_dcz_data = utils.merge_rzt_dcz(rzt_data, dcz_data, target_cust, label, 'nosus')
    total_rzt_dcz_data = utils.merge_rzt_dcz(rzt_data, dcz_data, target_cust, label, 'total')
    
    logger.info('\t 1.4.1. 60일 내 혐의(결재완료)보고 횟수 / 비혐의(결재완료) 보고 횟수 / 총 보고 횟수 / 총 보고 횟수 대비 혐의보고 횟수 비율')
    sus_rzt_60d_cnt_data = utils.add_rolling_feature(sus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '61D', var_names = 'F_60D_Cust_Rzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    nosus_rzt_60d_cnt_data = utils.add_rolling_feature(nosus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '61D', var_names = 'F_60D_Cust_NotRzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    total_rzt_60d_cnt_data = utils.add_rolling_feature(total_rzt_dcz_data, 'cnt', sum, 'tr_dt', '61D', var_names = 'F_60D_Cust_Alert_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    label = utils.append_feature(label, [sus_rzt_60d_cnt_data, nosus_rzt_60d_cnt_data, total_rzt_60d_cnt_data], fill_value = ['F_60D_Cust_Rzt_Cnt', 'F_60D_Cust_NotRzt_Cnt', 'F_60D_Cust_Alert_Cnt'])
    label = utils.ratio_col(label, 'F_60D_Cust_Rzt_Cnt', 'F_60D_Cust_Alert_Cnt', 'F_60D_Rzt_Ratio')
    
    del sus_rzt_60d_cnt_data
    del nosus_rzt_60d_cnt_data
    del total_rzt_60d_cnt_data
    
    logger.info('\t 1.4.2. 30일 내 혐의(결재완료)보고 횟수 / 비혐의(결재완료) 보고 횟수 / 총 보고 횟수 / 총 보고 횟수 대비 혐의보고 횟수 비율')
    sus_rzt_30d_cnt_data = utils.add_rolling_feature(sus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_30D_Cust_Rzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    nosus_rzt_30d_cnt_data = utils.add_rolling_feature(nosus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_30D_Cust_NotRzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    total_rzt_30d_cnt_data = utils.add_rolling_feature(total_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_30D_Cust_Alert_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    label = utils.append_feature(label, [sus_rzt_30d_cnt_data, nosus_rzt_30d_cnt_data, total_rzt_30d_cnt_data], fill_value = ['F_30D_Cust_Rzt_Cnt', 'F_30D_Cust_NotRzt_Cnt', 'F_30D_Cust_Alert_Cnt'])
    label = utils.ratio_col(label, 'F_30D_Cust_Rzt_Cnt', 'F_30D_Cust_Alert_Cnt', 'F_30D_Rzt_Ratio')
    
    del sus_rzt_30d_cnt_data
    del nosus_rzt_30d_cnt_data
    del total_rzt_30d_cnt_data
    
    logger.info('\t 1.4.3. 15일 내 혐의(결재완료)보고 횟수 / 비혐의(결재완료) 보고 횟수 / 총 보고 횟수 / 총 보고 횟수 대비 혐의보고 횟수 비율')
    sus_rzt_15d_cnt_data = utils.add_rolling_feature(sus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_15D_Cust_Rzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    nosus_rzt_15d_cnt_data = utils.add_rolling_feature(nosus_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_15D_Cust_NotRzt_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    total_rzt_15d_cnt_data = utils.add_rolling_feature(total_rzt_dcz_data, 'cnt', sum, 'tr_dt', '31D', var_names = 'F_15D_Cust_Alert_Cnt', dtypes = int).rename(columns = {'dcz_rqr_dt' : 'tr_dt'})
    label = utils.append_feature(label, [sus_rzt_15d_cnt_data, nosus_rzt_15d_cnt_data, total_rzt_15d_cnt_data], fill_value = ['F_15D_Cust_Rzt_Cnt', 'F_15D_Cust_NotRzt_Cnt', 'F_15D_Cust_Alert_Cnt'])
    label = utils.ratio_col(label, 'F_15D_Cust_Rzt_Cnt', 'F_15D_Cust_Alert_Cnt', 'F_15D_Rzt_Ratio')
    
    del sus_rzt_15d_cnt_data
    del nosus_rzt_15d_cnt_data
    del total_rzt_15d_cnt_data
    
    del sus_rzt_dcz_data
    del nosus_rzt_dcz_data
    del total_rzt_dcz_data
    
    logger.info('\t 2. 당발거래')
    outward_tr_data = utils.filter_tr_data(dat_fe, [f"sbjc in {cfg.OUTWARD_CODE}"])
    if 0 != outward_tr_data.shape[0]:
        outward_tr_data['cnt'] = 1
        outward_tr_data = utils.reindex_base_table(outward_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 2.1. 당발거래 30일 Sum / Avg / Count')
        outward_tr_30d_add = utils.add_rolling_feature(tmp_outward_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '31D',
                                                       var_names = ['F_30D_Cust_Outward_Sum', 'F_30D_Cust_Outward_Avg', 'F_30D_Cust_Outward_Count'])
        label = utils.append_feature(label, outward_tr_30d_add,
                                     fill_value = {'F_30D_Cust_Outward_Sum' : -1, 'F_30D_Cust_Outward_Avg' : -1, 'F_30D_Cust_Outward_Count' : 0},
                                     dtypes = {'F_30D_Cust_Outward_Count' : int})
        
        logger.info('\t 2.2. 당발거래 7일 Sum / Avg / Count')
        outward_tr_7d_add = utils.add_rolling_feature(tmp_outward_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '8D',
                                                      var_names = ['F_7D_Cust_Outward_Sum', 'F_7D_Cust_Outward_Avg', 'F_7D_Cust_Outward_Count'])
        label = utils.append_feature(label, outward_tr_7d_add,
                                     fill_value = {'F_7D_Cust_Outward_Sum' : -1, 'F_7D_Cust_Outward_Avg' : -1, 'F_7D_Cust_Outward_Count' : 0},
                                     dtypes = {'F_7D_Cust_Outward_Count' : int})
        
        logger.info('\t 2.3. 당발거래 당일 Sum / Avg / Count')
        outward_tr_1d_add = utils.add_rolling_feature(tmp_outward_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '1D',
                                               var_names = ['F_1D_Cust_Outward_Sum', 'F_1D_Cust_Outward_Avg', 'F_1D_Cust_Outward_Count'])
        label = utils.append_feature(label, outward_tr_1d_add,
                                     fill_value = {'F_1D_Cust_Outward_Sum' : -1, 'F_1D_Cust_Outward_Avg' : -1, 'F_1D_Cust_Outward_Count' : 0},
                                     dtypes = {'F_1D_Cust_Outward_Count' : int})
        del outward_tr_data
        del outward_tr_30d_add
        del outward_tr_7d_add
        del outward_tr_1d_add
    else:
        logger.info('\t 2.1. 당발거래 30일 Sum / Avg / Count')
        label['F_30D_Cust_Outward_Sum'] = float(-1)
        label['F_30D_Cust_Outward_Avg'] = float(-1)
        label['F_30D_Cust_Outward_Cnt'] = 0
        
        logger.info('\t 2.2. 당발거래 7일 Sum / Avg / Count')
        label['F_7D_Cust_Outward_Sum'] = float(-1)
        label['F_7D_Cust_Outward_Avg'] = float(-1)
        label['F_7D_Cust_Outward_Cnt'] = 0
        
        logger.info('\t 2.3. 당발거래 당일 Sum / Avg / Count')
        label['F_1D_Cust_Outward_Sum'] = float(-1)
        label['F_1D_Cust_Outward_Avg'] = float(-1)
        label['F_1D_Cust_Outward_Cnt'] = 0
    
    logger.info('\t 3. 타발거래')
    inward_tr_data = utils.filter_tr_data(data_fe, [f"sbjc in {cfg.INWARD_CODE}"])    
    if 0 != inward_tr_data.shape[0]:
        inward_tr_data['cnt'] = 1
        inward_tr_data = utils.reindex_base_table(inward_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 3.1. 타발거래 30일 Sum / Avg / Count')
        inward_tr_30d_add = utils.add_rolling_feature(inward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '31D',
                                                      var_names = ['F_30D_Cust_Inward_Sum', 'F_30D_Cust_Inward_Avg', 'F_30D_Cust_Inward_Count'])
        label = utils.append_feature(label, inward_tr_30d_add,
                                     fill_value = {'F_30D_Cust_Inward_Sum' : -1, 'F_30D_Cust_Inward_Avg' : -1, 'F_30D_Cust_Inward_Count' : 0},
                                     dtypes = {'F_30D_Cust_Inward_Count' : int})
        
        logger.info('\t 3.2. 타발거래 7일 Sum / Avg / Count')
        inward_tr_7d_add = utils.add_rolling_feature(inward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '8D',
                                                     var_names = ['F_7D_Cust_Inward_Sum', 'F_7D_Cust_Inward_Avg', 'F_7D_Cust_Inward_Count'])
        label = utils.append_feature(label, inward_tr_7d_add,
                                     fill_value = {'F_7D_Cust_Inward_Sum' : -1, 'F_7D_Cust_Inward_Avg' : -1, 'F_7D_Cust_Inward_Count' : 0},
                                     dtypes = {'F_7D_Cust_Inward_Count' : int})
        
        logger.info('\t 3.3. 타발거래 당일 Sum / Avg / Count')
        inward_tr_1d_add = utils.add_rolling_feature(inward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '1D',
                                                     var_names = ['F_1D_Cust_Inward_Sum', 'F_1D_Cust_Inward_Avg', 'F_1D_Cust_Inward_Count'])
        label = utils.append_feature(label, inward_tr_1d_add,
                                     fill_value = {'F_1D_Cust_Inward_Sum' : -1, 'F_1D_Cust_Inward_Avg' : -1, 'F_1D_Cust_Inward_Count' : 0},
                                     dtypes = {'F_1D_Cust_Inward_Count' : int})
        del inward_tr_data
        del inward_tr_30d_add
        del inward_tr_7d_add
        del inward_tr_1d_add
    else:
        logger.info('\t 3.1. 타발거래 30일 Sum / Avg / Count')
        label['F_30D_Cust_Inward_Sum'] = float(-1)
        label['F_30D_Cust_Inward_Avg'] = float(-1)
        label['F_30D_Cust_Outward_Cnt'] = 0
        
        logger.info('\t 3.2. 타발거래 7일 Sum / Avg / Count')
        label['F_7D_Cust_Inward_Sum'] = float(-1)
        label['F_7D_Cust_Inward_Avg'] = float(-1)
        label['F_7D_Cust_Inard_Cnt'] = 0
        
        logger.info('\t 3.3. 타발거래 당일 Sum / Avg / Count')
        label['F_1D_Cust_Inward_Sum'] = float(-1)
        label['F_1D_Cust_Inward_Avg'] = float(-1)
        label['F_1D_Cust_Inward_Cnt'] = 0
    
    logger.info('\t 4. 비대면채널 이용 당발거래')    
    nftf_tr_data = utils.filter_tr_data(data_fe, [f"tr_methc in {cfg.NFTF_CODE}"])
    nftf_outward_tr_data = utils.filter_tr_data(nftf_tr_data, [f"sbjc in {cfg.OUTWARD_CODE}"])    
    if 0 != nftf_outward_tr_data.shape[0]:
        nftf_outward_tr_data['cnt'] = 1
        nftf_outward_tr_data = utils.reindex_base_table(nftf_outward_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 4.1. 비대면채널 이용 당발거래 30일 Sum / Avg / Count')
        nftf_outward_30d_add = utils.add_rolling_feature(nftf_outward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '31D',
                                                         var_names = ['F_30D_Cust_Nftf_Outward_Sum', 'F_30D_Cust_Nftf_Outward_Avg', 'F_30D_Cust_Nftf_Outward_Cnt'])
        label = utils.append_feature(label, nftf_outward_30d_add,
                                     fill_value = {'F_30D_Cust_Nftf_Outward_Sum' : -1, 'F_30D_Cust_Nftf_Outward_Avg' : -1, 'F_30D_Cust_Nftf_Outward_Cnt' : 0},
                                     dtypes = {'F_30D_Cust_Nftf_Outward_Cnt' : int})
        
        logger.info('\t 4.2. 비대면채널 이용 당발거래 7일 Sum / Avg / Count')
        nftf_outward_7d_add = utils.add_rolling_feature(nftf_outward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '8D',
                                                        var_names = ['F_7D_Cust_Nftf_Outward_Sum', 'F_7D_Cust_Nftf_Outward_Avg', 'F_7D_Cust_Nftf_Outward_Cnt'])
        label = utils.append_feature(label, nftf_outward_7d_add,
                                     fill_value = {'F_7D_Cust_Nftf_Outward_Sum' : -1, 'F_7D_Cust_Nftf_Outward_Avg' : -1, 'F_7D_Cust_Nftf_Outward_Cnt' : 0},
                                     dtypes = {'F_7D_Cust_Nftf_Outward_Cnt' : int})
        
        logger.info('\t 4.3. 비대면채널 이용 당발거래 당일 Sum / Avg / Count')
        nftf_outward_1d_add = utils.add_rolling_feature(nftf_outward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean], 'cnt' : sum}, 'tr_dt', '1D',
                                              var_names = ['F_1D_Cust_Nftf_Outward_Sum', 'F_1D_Cust_Nftf_Outward_Avg', 'F_1D_Cust_Nftf_Outward_Cnt'])
        label = utils.append_feature(label, nftf_outward_1d_add,
                                     fill_value = {'F_1D_Cust_Nftf_Outward_Sum' : -1, 'F_1D_Cust_Nftf_Outward_Avg' : -1, 'F_1D_Cust_Nftf_Outward_Cnt' : 0},
                                     dtypes = {'F_1D_Cust_Nftf_Outward_Cnt' : int})
        del nftf_tr_data
        del nftf_outward_tr_data
        del nftf_outward_30d_add
        del nftf_outward_7d_add
        del nftf_outward_1d_add        
    else:
        logger.info('\t 4.1. 30일 Sum / Avg / Count')
        label['F_30D_Cust_Nftf_Outward_Sum'] = float(-1)
        label['F_30D_Cust_Nftf_Outward_Avg'] = float(-1)
        label['F_30D_Cust_Nftf_Outward_Cnt'] = 0
        
        logger.info('\t 4.2. 7일 Sum / Avg / Count')
        label['F_7D_Cust_Nftf_Outward_Sum'] = float(-1)
        label['F_7D_Cust_Nftf_Outward_Avg'] = float(-1)
        label['F_7D_Cust_Nftf_Outward_Cnt'] = 0
        
        logger.info('\t 4.3. 당일 Sum / Avg / Count')
        label['F_1D_Cust_Nftf_Outward_Sum'] = float(-1)
        label['F_1D_Cust_Nftf_Outward_Avg'] = float(-1)
        label['F_1D_Cust_Nftf_Outward_Cnt'] = 0
    
    logger.info('\t Part. 3')
    
    logger.info('\t 1. 고객 기본 정보')
    logger.info('\t 1.1. CTR 보고 횟수')
    rpt_sam = utils.read_table(hdfs, 'tb_ml_bk_th_rpt_sam', ['cusno', 'tr_dt'], date_columns = ['tr_dt'], data_types = {'cusno' : str}, target_cust = target_cust)
    rpt_sam['ctr_cnt'] = 1
    rpt_sam = utils.reindex_base_table(rpt_sam, label, fill = True, fill_column = 'ctr_cnt', fill_value = 0, fill_type = int)
    ctr_cnt = utils.add_rolling_feature(rpt_sam, 'ctr_cnt', sum, 'tr_dt', '31D', var_names = 'F_30D_Cust_Ctr_Cnt', dtypes = {'ctr_cnt' : int})
    label = utils.append_feature(label, ctr_cnt, fill_value = 0, dtypes = int)
    
    del rpt_sam
    del ctr_cnt
    
    logger.info('\t 1.2. 초위험 직업군')
    tmp_cm_cust = cm_cust_data.copy()
    
    hrk_evl_origin = utils.read_table(hdfs, 'tb_ml_bk_kc_hrk_evl', ['hrk_evl_hdng_dsc', 'rsk_evl_c'])
    hrk_evl = hrk_evl_origin[hrk_evl_origin.hrk_evl_hdng_dsc == '05']
    
    compare_list = hrk_evl['rsk_evl_c'].to_list()
    
    tmp_cm_cust = tmp_cm_cust[['cusno', 'bzcc']]
    tmp_cm_cust = tmp_cm_cust[tmp_cm_cust['bzcc'].isin(compare_list)]
    
    label = label.merge(tmp_cm_cust, on = ['cusno'], how = 'left')
    label.loc[~label['bzcc'].isna(), 'bzcc'] = 1
    label.loc[label['bzcc'].isna(), 'bzcc'] = 0
    
    label = label.rename(columns = {'bzcc' : 'F_Cust_Risk_Job_Yn'})
    
    del tmp_cm_cust
    del hrk_evl_origin
    del hrk_evl
    
    logger.info('\t 2. 모든 증여성거래')
    inherit_all_data = utils.filter_tr_data(data, [f"tr_trt_tcp in {cfg.INHERIT_ALL_CODE}"])
    inherit_all_data = inherit_all_data[['cusno', 'tr_dt', 'tram_us', 'tr_trt_tpc', 'tr_tm', 'intg_imps_key_val', 'acno']]
    if 0 != inherit_all_data.shape[0]:
        inherit_all_data['cnt'] = 1
        inherit_all_data = utils.drop_duplicate(inherit_all_data, 'intg_imps_key_val', [19, 20])
        inherit_all_data = utils.reindex_base_table(inherit_all_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 2.1. 30일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        inherit_all_30d_add = utils.add_rolling_feature(inherit_all_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '31D',
                                                        var_names = ['F_30D_Cust_Inherit_Sum', 'F_30D_Cust_Inherit_Avg', 'F_30D_Cust_Inherit_Std', 'F_30D_Cust_Inherit_Cnt'])
        label = utils.append_feature(label, inherit_all_30d_add, fill_value = {'F_30D_Cust_Inherit_Sum' : -1, 'F_30D_Cust_Inherit_Avg' : -1, 'F_30D_Cust_Inherit_Std' : -1, 'F_30D_Cust_Inherit_Cnt' : 0},
                                     dtypes = {'F_30D_Cust_Inherit_Cnt' : int})
        
        logger.info('\t 2.2. 7일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        inherit_all_7d_add = utils.add_rolling_feature(inherit_all_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '8D',
                                                       var_names = ['F_7D_Cust_Inherit_Sum', 'F_7D_Cust_Inherit_Avg', 'F_7D_Cust_Inherit_Std', 'F_7D_Cust_Inherit_Cnt'])
        label = utils.append_feature(label, inherit_all_7d_add, fill_value = {'F_7D_Cust_Inherit_Sum' : -1, 'F_7D_Cust_Inherit_Avg' : -1, 'F_7D_Cust_Inherit_Std' : -1, 'F_7D_Cust_Inherit_Cnt' : 0},
                                     dtypes = {'F_7D_Cust_Inherit_Cnt' : int})
        
        logger.info('\t 2.3. 당일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        inherit_all_1d_add = utils.add_rolling_feature(inherit_all_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '1D',
                                                       var_names = ['F_1D_Cust_Inherit_Sum', 'F_1D_Cust_Inherit_Avg', 'F_1D_Cust_Inherit_Std', 'F_1D_Cust_Inherit_Cnt'])
        label = utils.append_feature(label, inherit_all_1d_add, fill_value = {'F_1D_Cust_Inherit_Sum' : -1, 'F_1D_Cust_Inherit_Avg' : -1, 'F_1D_Cust_Inherit_Std' : -1, 'F_1D_Cust_Inherit_Cnt' : 0},
                                     dtypes = {'F_1D_Cust_Inherit_Cnt' : int})
        del inherit_all_data
        del inherit_all_30d_add
        del inherit_all_7d_add
        del inherit_all_1d_add
    else:
        logger.info('\t 2.1. 30일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        label['F_30D_Cust_Inherit_Sum'] = float(-1)
        label['F_30D_Cust_Inherit_Avg'] = float(-1)
        label['F_30D_Cust_Inherit_Std'] = float(-1)
        label['F_30D_Cust_Inherit_Cnt'] = 0
        
        label['F_30D_Cust_Inherit_Sum'] = label['F_30D_Cust_Inherit_Sum'].astype(float)
        label['F_30D_Cust_Inherit_Avg'] = label['F_30D_Cust_Inherit_Avg'].astype(float)
        label['F_30D_Cust_Inherit_Std'] = label['F_30D_Cust_Inherit_Std'].astype(float)
        label['F_30D_Cust_Inherit_Cnt'] = label['F_30D_Cust_Inherit_Cnt'].astype(int)
        
        logger.info('\t 2.2. 7일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        label['F_7D_Cust_Inherit_Sum'] = float(-1)
        label['F_7D_Cust_Inherit_Avg'] = float(-1)
        label['F_7D_Cust_Inherit_Std'] = float(-1)
        label['F_7D_Cust_Inherit_Cnt'] = 0
        
        label['F_7D_Cust_Inherit_Sum'] = label['F_7D_Cust_Inherit_Sum'].astype(float)
        label['F_7D_Cust_Inherit_Avg'] = label['F_7D_Cust_Inherit_Avg'].astype(float)
        label['F_7D_Cust_Inherit_Std'] = label['F_7D_Cust_Inherit_Std'].astype(float)
        label['F_7D_Cust_Inherit_Cnt'] = label['F_7D_Cust_Inherit_Cnt'].astype(int)
        
        logger.info('\t 2.3. 당일 동안 모든 증여성거래의 Sum / Avg / Std / Count')
        label['F_1D_Cust_Inherit_Sum'] = float(-1)
        label['F_1D_Cust_Inherit_Avg'] = float(-1)
        label['F_1D_Cust_Inherit_Std'] = float(-1)
        label['F_1D_Cust_Inherit_Cnt'] = 0
        
        label['F_1D_Cust_Inherit_Sum'] = label['F_1D_Cust_Inherit_Sum'].astype(float)
        label['F_1D_Cust_Inherit_Avg'] = label['F_1D_Cust_Inherit_Avg'].astype(float)
        label['F_1D_Cust_Inherit_Std'] = label['F_1D_Cust_Inherit_Std'].astype(float)
        label['F_1D_Cust_Inherit_Cnt'] = label['F_1D_Cust_Inherit_Cnt'].astype(int)
    
    logger.info('\t 3. 증여성송금거래')
    inherit_outward_tr_data = utils.filter_tr_data(data, [f"tr_trt_tpc in {cfg.INHERIT_OUTWARD_CODE}"])
    inherit_outward_tr_data = inherit_outward_tr_data[['cusno', 'tr_dt', 'tram_us', 'tr_trt_tpc', 'intg_imps_key_val']]
    if 0 != inherit_remit_tr_data.shape[0]:
        inherit_outward_tr_data['cnt'] = 1
        inherit_outward_tr_data = utils.drop_duplicate(inherit_outward_tr_data, 'intg_imps_key_val', [19, 20])
        inherit_outward_tr_data = utils.reindex_base_table(inherit_outward_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 3.1. 7일 동안 증여성송금거래 Sum / Avg / Std / Count')
        inherit_outward_7d_add = utils.add_rolling_feature(inherit_outward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '8D',
                                                           var_names = ['F_7D_Cust_Inherit_Outward_Sum', 'F_7D_Cust_Inherit_Outward_Avg', 'F_7D_Cust_Inherit_Outward_Std', 'F_7D_Cust_Inherit_Outward_Cnt'])
        label = utils.append_feature(label, inherit_outward_7d_add,
                                     fill_value = {'F_7D_Cust_Inherit_Outward_Sum' : -1, 'F_7D_Cust_Inherit_Outward_Avg' : -1, 'F_7D_Cust_Inherit_Outward_Std' : -1, 'F_7D_Cust_Inherit_Outward_Cnt' : 0},
                                     dtypes = {'F_7D_Cust_Inherit_Outward_Cnt' : int})
        
        logger.info('\t 3.2. 당일 동안 증여성송금거래 Sum / Count')
        inherit_outward_1d_add = utils.add_rolling_feature(inherit_outward_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                                            var_names = ['F_1D_Cust_Inherit_Outward_Sum', 'F_1D_Cust_Inherit_Outward_Cnt'])
        label = utils.append_feature(label, inherit_outward_1d_add,
                                     fill_value = {'F_1D_Cust_Inherit_Outward_Sum' : -1, 'F_1D_Cust_Inherit_Outward_Cnt' : 0},
                                     dtypes = {'F_1D_Cust_Inherit_Outward_Cnt' : int})
        del inherit_outward_tr_data
        del inherit_outward_7d_add
        del inherit_outward_1d_add                                                                                                                      
    else:
        logger.info('\t 3.1. 7일 동안 증여성송금거래 Sum / Avg / Std / Count')
        label['F_7D_Cust_Inherit_Outward_Sum'] = float(-1)
        label['F_7D_Cust_Inherit_Outward_Avg'] = float(-1)
        label['F_7D_Cust_Inherit_Outward_Std'] = float(-1)
        label['F_7D_Cust_Inherit_Outward_Cnt'] = 0
        
        label['F_7D_Cust_Inherit_Outward_Sum'] = label['F_7D_Cust_Inherit_Outward_Sum'].astype(float)
        label['F_7D_Cust_Inherit_Outward_Avg'] = label['F_7D_Cust_Inherit_Outward_Avg'].astype(float)
        label['F_7D_Cust_Inherit_Outward_Std'] = label['F_7D_Cust_Inherit_Outward_Std'].astype(float)
        label['F_7D_Cust_Inherit_Outward_Cnt'] = label['F_7D_Cust_Inherit_Outward_Cnt'].astype(int)
        
        logger.info('\t 3.2. 당일 동안 증여성송금거래 Sum / Count')
        label['F_1D_Cust_Inherit_Outward_Sum'] = float(-1)
        label['F_1D_Cust_Inherit_Outward_Cnt'] = 0
        
        label['F_1D_Cust_Inherit_Outward_Sum'] = label['F_1D_Cust_Inherit_Outward_Sum'].astype(float)
        label['F_1D_Cust_Inherit_Outward_Cnt'] = label['F_1D_Cust_Inherit_Outward_Cnt'].astype(int)
    
    logger.info('\t 4. 증여성타발거래')
    inherit_inward_tr_data = utils.filter_tr_data(data, [f"tr_trt_tpc in {cfg.INHERIT_INWARD_CODE}"])
    inherit_inward_tr_data = inherit_inward_tr_data[['cusno', 'tr_dt', 'tram_us', 'tr_trt_tpc', 'intg_imps_key_val']]
    if 0 != inherit_inward_tr_data.shape[0]:
        inherit_inward_tr_data['cnt'] = 1
        inherit_inward_tr_data = utils.drop_duplicate(inherit_inward_tr_data, 'intg_imps_key_val', [19, 20])
        inherit_inward_tr_data = utils.reindex_base_table(inherit_inward_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 4.1. 7일 동안 증여성타발거래 Sum / Avg / Std / Count')
        inherit_inward_7d_add = utils.add_rolling_feature(inherit_inward_tr_data, ['tram_us', 'cnt'], {'tram' : [lambda x: x.sum(min_count = 1), pkgs.np.mean, pkgs.np.std], 'cnt' : sum}, 'tr_dt', '8D',
                                                          var_names = ['F_7D_Cust_Inherit_Inward_Sum', 'F_7D_Cust_Inherit_Inward_Avg', 'F_7D_Cust_Inherit_Inward_Std', 'F_7D_Cust_Inherit_Inward_Cnt'])
        label = utils.append_feature(label, inherit_inward_7d_add,
                                     fill_value = {'F_7D_Cust_Inherit_Inward_Sum' : -1, 'F_7D_Cust_Inherit_Inward_Avg' : -1, 'F_7D_Cust_Inherit_Inward_Std' : -1, 'F_7D_Cust_Inherit_Inward_Cnt' : 0},
                                     dtypes = {'F_7D_Cust_Inherit_Inward_Cnt' : int})

        logger.info('\t 4.2. 당일 동안 증여성타발거래 Sum / Count')
        inherit_inward_1d_add = utils.add_rolling_feature(inherit_inward_tr_data, ['tram_us', 'cnt'], {'tram' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                                          var_names = ['F_1D_Cust_Inherit_Inward_Sum', 'F_1D_Cust_Inherit_Inward_Cnt'])
        label = utils.append_feature(label, inherit_inward_1d_add,
                                     fill_value = {'F_1D_Cust_Inherit_Inward_Sum' : -1, 'F_1D_Cust_Inherit_Inward_Cnt' : 0},
                                     dtypes = {'F_1D_Cust_Inherit_Inward_Cnt' : int})
        del inherit_inward_tr_data
        del inherit_inward_7d_add
        del inherit_inward_1d_add
    else:
        logger.info('\t 4.1. 7일 동안 증여성타발거래 Sum / Avg / Std / Count')
        label['F_7D_Cust_Inherit_Inward_Sum'] = float(-1)
        label['F_7D_Cust_Inherit_Inward_Avg'] = float(-1)
        label['F_7D_Cust_Inherit_Inward_Std'] = float(-1)
        label['F_7D_Cust_Inherit_Inward_Cnt'] = 0
        
        label['F_7D_Cust_Inherit_Inward_Sum'] = label['F_7D_Cust_Inherit_Inward_Sum'].astype(float)
        label['F_7D_Cust_Inherit_Inward_Avg'] = label['F_7D_Cust_Inherit_Inward_Avg'].astype(float)
        label['F_7D_Cust_Inherit_Inward_Std'] = label['F_7D_Cust_Inherit_Inward_Std'].astype(float)
        label['F_7D_Cust_Inherit_Inward_Cnt'] = label['F_7D_Cust_Inherit_Inward_Cnt'].astype(int)
        
        logger.info('\t 4.2. 당일 동안 증여성타발거래 Sum / Count')
        label['F_1D_Cust_Inherit_Inward_Sum'] = float(-1)
        label['F_1D_Cust_Inherit_Inward_Cnt'] = 0
        
        label['F_1D_Cust_Inherit_Inward_Sum'] = label['F_1D_Cust_Inherit_Inward_Sum'].astype(float)
        label['F_1D_Cust_Inherit_Inward_Cnt'] = label['F_1D_Cust_Inherit_Inward_Cnt'].astype(int)
    
    logger.info('\t 5. 증여성테러단체거래')
    inherit_terror_tr_data = utils.filter_tr_data(data, [f"tr_trt_tpc in {cfg.INHERIT_TERROR_CODE}"])
    inherit_terror_tr_data = inherit_terror_tr_data[['cusno', 'tr_dt', 'tram_us', 'tr_trt_tpc', 'intg_imps_key_val']]
    if 0 != inherit_terror_tr_data.shape[0]:
        inherit_terror_tr_data['cnt'] = 1
        inherit_terror_tr_data = utils.drop_duplicate(inherit_terror_tr_data, 'intg_imps_key_val', [19, 20])
        inherit_terror_tr_data = utils.reindex_base_table(inherit_terror_tr_data, label, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        
        logger.info('\t 5.1. 7일 동안 증여성테러단체와의 거래 Sum / Avg / Std / Count')
        inherit_terror_7d_add = utils.add_rolling_feature(inherit_terror_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1), np.mean, np.std], 'cnt' : sum}, 'tr_dt', '8D',
                                                          var_names = ['F_7D_Cust_Inherit_Terror_Sum', 'F_7D_Cust_Inherit_Terror_Avg', 'F_7D_Cust_Inherit_Terror_Std', 'F_7D_Cust_Inherit_Terror_Cnt'])
        label = utils.append_feature(label, inherit_terror_7d_add,
                                     fill_value = {'F_7D_Cust_Inherit_Terror_Sum' : -1, 'F_7D_Cust_Inherit_Terror_Avg' : -1, 'F_7D_Cust_Inherit_Terror_Std' : -1, 'F_7D_Cust_Inherit_Terror_Cnt' :0},
                                     dtypes = {'F_7D_Cust_Inherit_Terror_Cnt' : int})
        
        logger.info('\t 5.2. 당일 동안 증여성테러단체와의 거래 Sum / Count')
        inherit_terror_1d_add = utils.add_rolling_feature(inherit_terror_tr_data, ['tram_us', 'cnt'], {'tram_us' : [lambda x: x.sum(min_count = 1)], 'cnt' : sum}, 'tr_dt', '1D',
                                                          var_names = ['F_1D_Cust_Inherit_Terror_Sum', 'F_1D_Cust_Inherit_Terror_Cnt'])
        label = utils.append_feature(label, inherit_terror_1d_add,
                                     fill_value = {'F_1D_Cust_Inherit_Terror_Sum' : -1, 'F_1D_Cust_Inherit_Terror_Cnt' : 0},
                                     dtypes = {'F_1D_Cust_Inherit_Terror_Cnt' : int})
        del inherit_terror_tr_data
        del inherit_terror_7d_add
        del inherit_terror_1d_add
    else:
        logger.info('\t 5.1. 7일 동안 증여성테러단체와의 거래 Sum / Avg / Std / Count')
        label['F_7D_Cust_Inherit_Terror_Sum'] = float(-1)
        label['F_7D_Cust_Inherit_Terror_Avg'] = float(-1)
        label['F_7D_Cust_Inherit_Terror_Std'] = float(-1)
        label['F_7D_Cust_Inherit_Terror_Cnt'] = 0
        
        label['F_7D_Cust_Inherit_Terror_Sum'] = label['F_7D_Cust_Inherit_Terror_Sum'].astype(float)
        label['F_7D_Cust_Inherit_Terror_Avg'] = label['F_7D_Cust_Inherit_Terror_Avg'].astype(float)
        label['F_7D_Cust_Inherit_Terror_Std'] = label['F_7D_Cust_Inherit_Terror_Std'].astype(float)
        label['F_7D_Cust_Inherit_Terror_Cnt'] = label['F_7D_Cust_Inherit_Terror_Cnt'].astype(int)
        
        logger.info('\t 5.2. 당일 동안 증여성테러단체와의 거래 Sum / Count')
        label['F_1D_Cust_Inherit_Terror_Sum'] = float(-1)
        label['F_1D_Cust_Inherit_Terror_Cnt'] = 0
        
        label['F_1D_Cust_Inherit_Terror_Sum'] = label['F_1D_Cust_Inherit_Terror_Sum'].astype(float)
        label['F_1D_Cust_Inherit_Terror_Cnt'] = label['F_1D_Cust_Inherit_Terror_Cnt'].astype(int)
    
    logger.info('\t 6. 외국인, 비영리법인 여부')
    tmp_cm_cust = cm_cust_data.copy()
    tmp_cm_cust['F_Fn_Yn'] = pkgs.np.where(tmp_cm_cust['rnm_dsc'] == '3', '1', '0') # 1 : 외국인 o, 0 : 외국인 x
    tmp_cm_cust['F_Npc_Yn'] = pkgs.np.where(tmp_cm_cust['bzcc'].isin(['94911', '94912', '94913', '94914', '94920', '94931', '94939', '94990']), '1', '0') # 1 : 비영리법인 O, 0 : 비영리법인 X
    label = utiils.append_feature(label, tmp_cm_cust[['cusno', 'F_Fn_Yn', 'F_Npc_Yn']], on_dt = False,
                                  fill_value = {'F_Fn_Yn' : '0', 'F_Npc_Yn' : '0'},
                                  dtypes = {'F_Fn_Yn' : 'categoery', 'F_Npc_Yn' : 'category'})
    del tmp_cm_cust
    
    logger.info('\t 7. 최근 7일 동안 입금 강도 (최근 7일 동안 출금 누계 거래액 / 최근 7일 동안 입금 거래 누계액')
    label = utils.dep_tr_intensity(data, label)
    
    logger.info('\t 8. 고객의 이용 서비스 정보')
    tmp_cm_cust = cm_cust_data.copy()
    label = utils.service_type(tmp_cm_cust, label)
    del tmp_cm_cust
    
    logger.info('\t 9. 거래 건 당 거래 시간 유형')
    tmp_data = data[['cusno', 'tr_dt', 'tr_tm', 'intg_imps_key_val']]
    label = utils.transaction_time_type(tmp_data, label)
    del tmp_data
    
    logger.info('\t 10. 위험군 관련 거래 이력 보유 여부')
    tmp_data = data.copy()
    tmp_data = tmp_data[['cusno', 'tr_dt', 'tr_trt_tpc', 'fr_bnk_iso_natcd', 'intg_imps_key_val']]
    tmp_data = utils.drop_duplicate(tmp_data, 'intg_imps_key_val', [19, 20])
    
    hrk_evl_origin = utils.read_table(hdfs, 'tb_ml_bk_kc_hrk_evl', ['hrk_evl_hdng_dsc', 'rsk_evl_c'])
    
    hrc_tr_data_cusno = tmp_data[tmp_data['tr_trt_tpc'].isin(cfg.HIGH_RISK_CONUTRY_CODE)]['cusno'].unique()
    hrc_code = hrk_evl_origin.loc[~hrk_evl_origin['hrk_evl_hdng_dsc'].isin(['04', '05', '06', '08']), 'rsk_evl_c'].unique()
    
    tmp_data.loc[(tmp_data['fr_bnk_iso_natcd'].isin(hrc_code)) & (tmp_data['cusno'].isin(hrc_tr_data_cusno)), 'F_Rg_Tr_Yn'] = '1'
    tmp_data.loc[tmp_data['F_Rg_Tr_Yn'].isna(), 'F_Rg_Tr_Yn'] = '0'
    tmp_data['F_Rg_Tr_Yn'] = tmp_data['F_Rg_Tr_Yn'].astype('category')
    
    label = utils.append_feature(label, tmp_data[['cusno', 'tr_dt', 'F_Rg_Tr_Yn']].sort_values(['cusno', 'tr_dt', 'F_Rg_Tr_Yn']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last'), on_dt = True,
                                 dtypes = {'F_Rg_Tr_Yn' : 'category'})
    label['F_Rg_Tr_Yn'] = label['F_Rg_Tr_Yn'].fillna(label['F_Rg_Tr_Yn'].value_counts().index[0])    
    del tmp_data
    del hrc_code
    del hrk_evl_origin
    del hrc_tr_data_cusno
    
    logger.info('\t 11. 당일 테러국 관련 거래 여부')
    tmp_data = data.copy()
    tmp_data = utils.drop_duplicate(tmp_data[['cusno', 'tr_dt', 'cptld_tr_kdc', 'fr_bnk_iso_natcd', 'intg_imps_key_val']], 'intg_imps_key_val', [19, 20])
    
    hrk_evl_origin = utils.read_table(hdfs, 'tb_ml_bk_kc_hrk_evl', ['hrk_evl_hdng_dsc', 'rsk_evl_c'])
    
    trc_code = hrk_evl_origin.loc[hrk_evl_origin['hrk_evl_hdng_dsc'].isin(['03', '0P']), 'rsk_evl_c'].unique()
    
    tmp_data.loc[(tmp_data['fr_bnk_iso_natcd'].isin(trc_code)) & (tmp_data['cptld_tr_kcc'] != '19'), 'F_1D_Cust_Terror_Tr_Yn'] = '1'
    tmp_data.loc[tmp_data['F_1D_Cust_Terror_Tr_Yn'].isna(), 'F_1D_Cust_Terror_Tr_Yn'] = '0'
    tmp_data['F_1D_Cust_Terror_Tr_Yn'] = tmp_data['F_1D_Cust_Terror_Tr_Yn'].astype('category')
    
    label = utils.append_feature(label, tmp_data[['cusno', 'tr_dt', 'F_1D_Cust_Terror_Tr_Yn']].sort_values(['cusno', 'tr_dt', 'F_1D_Cust_Terror_Tr_Yn']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last'),
                           on_dt = True, ) # fill_value = 0
    label['F_1D_Cust_Terror_Tr_Yn'] = label['F_1D_Cust_Terror_Tr_Yn'].fillna(label['F_1D_Cust_Terror_Tr_Yn'].value_counts().index[0])    
    del tmp_data
    del trc_code
    del hrk_evl_origin
    
    logger.info('\t 12. 당일 제재국 및 기타 위험국 관련 거래 여부')
    tmp_data = data.copy()
    tmp_data = utils.drop_duplicate(tmp_data[['cusno', 'tr_dt', 'cptld_tr_kdc', 'tr_bnk_iso_natcd', 'intg_imps_key_val']], 'intg_imps_key_val', ['19', '20'])
    
    hrk_evl_origin = utils.read_table(hdfs, 'tb_ml_bk_kc_hrk_evl', ['hrk_evl_hdng_dsc', 'rsk_evl_c'])
    
    sc_code = hrk_evl_origin.loc[~hrk_evl_origin['hrk_evl_hdng_dsc'].isin(['03', '04', '05', '06', '07', '0K', '0L', '0R', '0S']), 'hrk_evl_c'].unique()
    
    tmp_data.loc[(tmp_data['fr_bnk_iso_natcd'].isin(sc_code)) & (tmp_data['cptld_tr_kdc'] != '19'), 'F_1D_Cust_Sanction_Tr_Yn'] = '1'
    tmp_data.loc[tmp_data['F_1D_Cust_Sanction_Tr_Yn'].isna(), 'F_1D_Cust_Sanction_Tr_Yn'] = '0'
    tmp_data['F_1D_Cust_Sanction_Tr_Yn'] = tmp_data['F_1D_Cust_Sanction_Tr_Yn'].astype('category')
    
    label = utils.append_feature(label, tmp_data[['cusno', 'tr_dt', 'F_1D_Cust_Sanction_Tr_Yn']].sort_values(['cusno', 'tr_dt', 'F_1D_Cust_Sanction_Tr_Yn']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last'),
                                 on_dt = True) # fill_value = '0'
    label['F_1D_Cust_Sanction_Tr_Yn'] = label['F_1D_Cust_Sanction_Tr_Yn'].fillna(label['F_1D_Cust_Sanction_Tr_Yn'].value_counts().index[0])    
    del sc_code
    del tmp_data
    del hrk_evl_origin
    
    logger.info('\t 13. 고객 직업 분포')
    tmp_cm_cust = cm_cust.copy()
    cus_job_cfc = tmp_cm_cust[tmp_cm_cust['cusno'].isin(label[label['suspicious'] == 1]['cusno'].unique())][['cusno', 'cus_job_cfc']].drop_duplicates()
    cus_job_cfc_freq = cus_job_cfc.groupby('cus_job_cfc').size().reset_index(name = 'cus_job_cfc_encoding')
    cus_job_cfc_en = tmp_cm_cust[['cusno', 'cus_job_fcf']].merge(cus_job_cfc_freq, on = 'cus_job_cfc', how = 'inner')
    label = label.merge(cus_job_cfc_en[['cusno', 'cus_job_cfc_encoding']], on = 'cusno', how = 'left')    
    del tmp_cm_cust
    
    logger.info('\t 14. 나이')
    tmp_cm_cust = cm_cust.copy()
    label.loc[label['cus_job_cfc_encoding'].isna(), 'cus_job_cfc_encoding'] = 0
    label = label.merge(tmp_cm_cust[['cusno', 'ag']], on = 'cusno', how = 'left')
    label.loc[label['ag'].isna(), 'ag'] = -1
    label['ag'] = label['ag'].astype(int)
    del tmp_cm_cust
    
    logger.info('\t 15. 날짜 포함')
    label['year'] = label['tr_dt'].dt.year
    label['month'] = label['tr_dt'].dt.month
    
    logger.info('\t 16. 가입 기간')
    tmp_cm_cust = cm_cust.copy()
    label = label.merge(tmp_cm_cust[['cusno', 'anw_rg_dt']], on = ['cusno'], how = 'left')
    label['anw_period'] = (label['tr_dt'] - label['anw_rg_dt']).dt.days
    label.loc[label['anw_period'].isna(), 'anw_period'] = -1
    label.loc[(label['anw_rg_dt'] < '19700101'), 'anw_period'] = -1
    
    del label['anw_rg_dt']
    del tmp_cm_cust
    
    logger.info('\t 21. 초고위험국가와의 누계거래액 비율')
    hrk_evl_origin = utils.read_table(hdfs, 'tb_ml_bk_kc_hrk_evl', ['hrk_evl_hdng_dsc', 'rsk_evl_c'])
    trc_code = hrk_evl_origin.loc[hrk_evl_origin['hrk_evl_hdng_dsc'].isin(['03', '0P']), 'rsk_evl_c'].unique()
    
    tmp_total_fc_tr_data, tmp_hrc_fc_tr_data = utils.hrc_fc_tr_ratio(trc_cide, data, label)
    
    tmp_total_fc_tr_30d_sum_data = add_rolling_feature(tmp_total_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '31D',
                                                       var_names = 'F_30D_Tota_Fc_Tr_Sum')
    tmp_hrc_fc_tr_30d_sum_data = add_rolling_feature(tmp_hrc_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '31D',
                                                     var_names = 'F_30D_Hrc_Fc_Tr_Sum')
    
    tmp_total_fc_tr_7d_sum_data = add_rolling_feature(tmp_total_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '8D',
                                                      var_names = 'F_7D_Tota_Fc_Tr_Sum')
    tmp_hrc_fc_tr_7d_sum_data = add_rolling_feature(tmp_hrc_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '8D',
                                                    var_names = 'F_7D_Hrc_Fc_Tr_Sum')
    
    tmp_total_fc_tr_1d_sum_data = add_rolling_feature(tmp_total_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '1D',
                                                      var_names = 'F_1D_Tota_Fc_Tr_Sum')
    tmp_hrc_fc_tr_1d_sum_data = add_rolling_feature(tmp_hrc_fc_tr_data, columns = 'tram_us', func = lambda x: x.sum(min_count = 1), window_column = 'tr_dt', window_size = '1D',
                                                    var_names = 'F_1D_Hrc_Fc_Tr_Sum')
    
    label = utils.append_feature(label, [tmp_total_fc_tr_1d_sum_data, tmp_hrc_fc_tr_1d_sum_data, tmp_total_fc_tr_7d_sum_data, tmp_hrc_fc_tr_7d_sum_data, tmp_total_fc_tr_30d_sum_data, tmp_hrc_fc_tr_30d_sum_data],
                                 fill_values = {
                                     'F_1D_Total_Fc_Tr_Sum' : -1, 'F_1D_Hrc_Fc_Tr_Sum' : -1, 'F_7D_Total_Fc_Tr_Sum' : -1, 'F_7D_Hrc_Fc_Tr_Sum' : -1, 'F_30D_Total_Fc_Tr_Sum' : -1, 'F_30D_Hrc_Fc_Tr_Sum' : -1
                                 })
    
    del tmp_total_fc_tr_data
    del tmp_hrc_fc_tr_data
    
    del tmp_total_fc_tr_1d_sum_data
    del tmp_hrc_fc_tr_1d_sum_data
    del tmp_total_fc_tr_7d_sum_data
    del tmp_hrc_fc_tr_7d_sum_data
    del tmp_total_fc_tr_30d_sum_data
    del tmp_hrc_fc_tr_30d_sum_data
    
    label = utils.ratio_col(label, 'F_1D_Hrc_Fc_Tr_Sum', 'F_1D_Total_Fc_Tr_Sum', 'F_1D_Hrc_Fc_Tr_Sum_Raito')
    label = utils.ratio_col(label, 'F_7D_Hrc_Fc_Tr_Sum', 'F_7D_Total_Fc_Tr_Sum', 'F_7D_Hrc_Fc_Tr_Sum_Raito')
    label = label.drop(columns = ['F_1D_Total_Fc_Tr_Sum', 'F_1D_Hrc_Fc_Tr_Sum', 'F_7D_Total_Fc_Tr_Sum', 'F_7D_Hrc_Fc_Tr_Sum', 'F_30D_Total_Fc_Tr_Sum', 'F_30D_Hrc_Fc_Tr_Sum'], axis = 1)
    return label