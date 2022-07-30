import packages as pkgs
import config as cfg

###########################
### DATA LOAD FUNCTIONS ###
###########################
def read_hive_table(hdfs, table_name):
    file_loc = f'/apps/hive/warehouse/nbbpsrc.db/{table_name}'
    
    file_list = hdfs.glob(pkgs.os.path.join(file_loc, '*'))
    file_list = list(filter(lambda x: 'hive-staging' not in x, file_list))
    if 0 == len(file_list):
        pass
    elif 2 == len(file_list[0].split('/').split('=')):
        sub_file_list = hdfs.glob(pkgs.os.path.join(file_loc, '**/*'))
    else:
        sub_file_list = hdfs.glob(pkgs.os.path.join(file_loc, '*.orc'))
    sub_file_list = list(filter(lambda x: 'hive-staging' not in x, sub_file_list))
    return sub_file_list

def get_parquet_file_list(hdfs, file_paht):
    file_list = hdfs.glob(pkgs.os.path.join(file_paht, '*.parquet'))
    if 0 == len(file_list):
        pass
    return file_list

def read_table(hdfs, table_name, columns = None, data_types = None, date_columns = None, datetime_columns = None, is_hive = True, target_cust = None):
    table_df_list = []
    
    if is_hive:
        for file in read_hive_table(hdfs, table_name):
            with hdfs.open(file) as f:
                if columns is not None:
                    table_df_list.append(pkgs.pd.read_orc(f, columns = columns))
                else:
                    table_df_list.append(pkgs.pd.read_orc(f))
    else:
        for file in get_parquet_file_list(hdfs, table_name):
            if columns is not None:
                table_df_list.append(pkgs.pd.read_parquet(f, columns = columns))
            else:
                table_df_list.append(pkgs.pd.read_parquet(f))
    
    if data_types is not None:
        df = df.astype(data_types)
    if date_columns is not None:
        df[date_columns] = df[date_columns].apply(pkgs.pd.to_datetime, format = '%Y-%m-%d', errors = 'coerce')
    if pkgs.datetime is not None:
        df[datetime_columns] = df[datetime_columns].apply(pkgs.pd.to_datetime, format = '%Y%m%d:%H:%M:%S', errors = 'coerce')
    
    for table in table_df_list:
        del table
    
    del table_df_list
    
    if target_cust is not None:
        df = df[df['cusno'].isin(target_cust)].reset_index(drop = True)
    return df

############################
### PREPROCESS FUNCTIONS ###
############################
def reindex_base_table(base, label, base_dt = 'tr_dt', label_dt = 'tr_dt', standard = 'cusno', fill = False, fill_column = None, fill_value = None, fill_type = None):
    # label 기준으로 rolling 가능하도록 outer join
    if 'cusno' == standard:
        reindex_df = base.merge(label[['cusno', label_dt]], left_on = ['cusno', base_dt], right_on = ['cusno', label_dt], how = 'outer')
    elif 'acno' == standard:
        tmp_label = label[['cusno', 'tr_dt']].merge(base.drop_dupliceats(['cusno', 'acno'])[['cusno', 'acno']], on = 'cusno', how = 'left').dropna(subset = ['acno'])
        reindex_df = base.merge(tmp_label[['cusno', 'acno', label_dt]], left_on = ['cusno', 'acno', base_dt], right_on = ['cusno', 'acno', label_dt], how = 'outer')
    else:
        pass
    
    # 빠진 row에 값 채우기 & type casting
    if fill:
        reindex_df.loc[reindex_df[fill_column].isna(), fill_column] = fill_value
        reindex_df[fill_column] = reindex_df[fill_column].astype(fill_type)
    return reindex_df

def add_rolling_feature(base, columns, func, window_column, window_size, group_columns = 'cusno', var_names = None, dtypes = None, final = 'last', agg_cusno = True):
    columns = [columns] if not isinstance(var_names, list) else columns
    group_columns = [group_columns] if not isinstance(group_columns, list) else group_columns
    
    final_pipe = pd.core.groupby.GroupBy.last if 'last' == final else pd.core.groupby.GroupBy.max
    
    if isinstance(func, dict):
        columns = list(func.keys())
    
    if '1D' == window_size:
        group_df = base.groupby(group_columns + [window_column])
        
        if 'nunique' == func: # nunique는 agg에는 느림
            feature = group_df[columns].nunique().astype(dtypes)
        else:
            feature = group_df[columns].agg(func).astype(dtypes)
    else:
        group_rolling_df = base.set_index(window_column).sort_index().groupby(group_columns)[columns].rolling(window_size, min_periods = 1)
        feature = group_rolling_df.agg(func).astype(dtypes)
    
    if 'acno' not in group_columns:
        feature = feature.groupby(level = [0, 1]).pipe(final_pipe)
    else: # gorup_columns = ['cusno', 'acno']
        feature = feature.groupby(level = [0, 1, 2]).pipe(final_pipe)
        if agg_cusno:
            feature = feature.groupby(level = [0, 2]).pipe(final_pipe)
    
    if var_names:
        var_names = [var_names] if not isinstance(var_names, list) else varl_names
        feature.columns = var_names
    
    feature = feature.reset_index()
    return feature

def append_feature(label, features, on_dt = True, label_dt = 'tr_dt', feature_dt = 'tr_dt', fill_value = None, dtypes = None):
    if fill_value is not None or dtypes:
        new_features = set()
    
    if not isinstance(features, list):
        features = [features]
    
    for feature in features:
        if on_dt:
            label = label.merge(feature, left_on = ['cusno', label_dt], right_on = ['cusno', feature_dt], how = 'left')
        else:
            label = label.merge(feature, on = 'cusno', how = 'left')
        
        if label_dt != feature_dt:
            label = label.drop(feature_dt, axis = 1)
        
        if  fill_value is not None or dtypes:
            new_feature_tmp = list(set(feature.columns) - set(['cusno', feature_dt]))
            new_feature = new_feature.union(new_feature_tmp)
    
    if fill_value is not None or dtypes:
        new_feature = list(new_feature)
    
    if fill_value is not None:
        label[new_feature] = label[new_feature].fillna(fill_value)
    
    if dtypes:
        label[new_feature] = label[new_feature].astype(dtypes)
    return label

def filter_tr_data(data, expr_list, columns = None):
    expr_str = ' and '.join(expr_list)
    if not columns:
        columns = data.columns
    return data.loc[data.eval(expr_str), columns].reset_index(drop = True)

def drop_duplicate(table, target, col_length):
    new_table = table.copy()
    new_table['tmp_key'] = new_table[target].astype(str)
    new_table['len_key'] = new_table[target].apply(lambda x: len(x))
    new_table['tmp_key'] = new_table['tmp_key'].apply(lambda x: x[:-3])
    
    empty = new_table[new_table['intg_imps_key_val'].isna()] # intg_imps_key_val is NaN
    not_empty = new_table[~new_table['intg_imps_key_val'].isna()] # intg_imps_key_val is not NaN
    columns = not_empty.columns
    target = [f"len_key not in {col_length}"]
    expr_str = ' and '.join(target)
    not_target = not_empty.loc[not_empty.eval(expr_str), columns].reset_index(drop = True)
    
    # Focus on target for drop_duplicates
    total_select = [empty, not_target]
    for i in col_length:
        new_i = [i]
        select_str = f"len_key in {new_i}"
        target_df = not_empty.loc[not_empty.eval(select_str), columns].reset_index(drop = True)
        target_df = target_df.drop_duplicates(['cusno', 'tr_dt', 'tmp_key']).sort_values(['cusno', 'tr_dt'])
        total_select.append(target_df)
    final_df = pkgs.pd.concat(total_select, axis = 0).sort_values(['cusno', 'tr_dt']).reset_index(drop = True)
    return final_df

def matching_trlog_tr_date(rzt_label_date, trlog_data, feature, days):
    # Rolling하기 위해 rzt_label_dat의 거래일(tr_dt) 이전 n일을 추가하기 위한 과정
    # N일을 추가해 trlog에 동일한 tr_dt에 해당하는 feature들의 정보를 가져와서 merge
    
    if 1 == days:
        rzt_tr = rzt_label_date[['cusno', 'tr_dt']].merge(trlog_data[['cusno', 'tr_dt'] + feature], on = ['cusno', 'tr_dt'], how = 'left') # rzt_label_date의 tr_dt는 label_df의 tr_dt와 동일
    else:
        rzt_label_date['tr_st_dt'] = rzt_label_date['tr_dt'] - timedelta(days = days) # EX) 3일동안이면 days = 2, 7일동안이면 days = 6, 30일동안이면 days = 29
        rzt_label_date = rzt_label_date.rename(columns = {'tr_dt' : 'tr_ed_dt'})[['cusno', 'tr_st_dt', 'tr_ed_dt']]

        range_label = rzt_label_date.apply(lambda x: pd.date_range(start = x['tr_st_dt'], end = x['tr_ed_dt'], freq = '1D').tolist(), axis = 1)

        rzt_label_date['tr_dt'] = range_label
        rzt_label_date = rzt_label_date.explode('tr_dt').drop_duplicates(['cusno', 'tr_dt'])
        rzt_tr = rzt_label_date[['cusno', 'tr_dt']].merge(trlog_data[['cusno', 'tr_dt'] + feature], on = ['cusno', 'tr_dt'], how = 'left')
    return rzt_tr

def change_to_category(raw):
    try:
        value = str(raw)
        value = value.replace('/', '')
        value = value.replace('.', '')
        value = value.replace(' ', '')
        value = value.strip()
        return str(value)
    except:
        return str(raw)

def handling_acno_type(table, columns, new_name):
    new_table = table.copy()
    for i in columns:
        name = new_name.get(i)
        new_table['tmp' + i] = new_table[i].apply(change_to_category)
        new_table[name] = pd.Categorical(new_table['tmp' + i]).codes
        new_table[name] = new_table[name].astype(int)
        
        del new_table['tmp' + i]
    return new_table

def change_to_category_with_none(raw):
    try:
        if None == raw:
            return 'unkown'
        else:
            value = str(raw)
            value = value.replace('/', '')
            value = value.replace('.', '')
            value = value.replace(' ', '')
            return str(value)
    except:
        return str(raw)

def handling_acno_type_with_none(table, columns, new_name):
    new_table = table.copy()
    for i in columns:
        name = new_name.get(i)
        new_table['tmp'] = new_table[i].apply(change_to_category_with_none)
        new_table[name] = pd.Categorical(new_table['tmp']).codes
        new_table[name] = new_table[name].astype(int)
        
        del new_table['tmp']
    return new_table

def ratio_col(label, first_col, second_col, out_col):
    tmp = label[[first_col, second_col]]
    
    if '_Cnt' in first_col:
        tmp.loc[tmp[second_col] == 0, second_col] = np.nan
        label[out_col] = (tmp[first_col] / tmp[second_col])
        label.loc[label[out_col].isna(), out_col] = float(0)
    else:
        tmp.loc[tmp[first_col] == -1, first_col] = np.nan
        tmp.loc[tmp[second_col] == -1, second_col] = np.nan
        label[out_col] = (tmp[first_col] / tmp[second_col])
        label.loc[label[out_col].isna(), out_col] = float(-1)
    return label

def value_select(table):
    n_table = table[~table.isna()]
    unique_acc = n_table.unique()
    return unique_acc.shape[0]

def merge_rzt_dcz(rzt_data, dcz_data, target_cust, label_data, div):
    # SSPN_TR_STSC ('S' : 혐의보고, 'Z' : 기혐의처리, 'N' : 비혐의, 'Y' : 기비혐의처리)
    # DCZ_SQNO ('5' : 결재 완료)
    
    if 'sus' == div:
        merge_data = rzt_data[rzt_data['sspn_tr_stsc'].isin(['S', 'Z'])].merge(dcz_data[dcz_data['now_sts_dsc'] == '5'], on = 'dcz_sqno', how = 'inner')
        merge_data['cnt'] = 1
        merge_data = reindex_base_table(base = merge_data.drop('tr_dt', axis = 1), label = label_data, base_dt = 'dcz_rqr_dt', label_dt = 'tr_dt', fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        merge_data.loc[merge_data['tr_dt'].isna(), 'tr_dt'] = merge_data.loc[merge_data['tr_dt'].isna(), 'dcq_rqr_dt']
    elif 'nosus' == div:
        merge_data = rzt_data[rzt_data['sspn_tr_stsc'].isin(['N', 'Y'])].merge(dcz_data[dcz_data['now_sts_dsc'] == '5'], on = 'dcz_sqno', how = 'inner')
        merge_data['cnt'] = 1
        merge_data = reindex_base_table(base = merge_data.drop('tr_dt', axis = 1), label = label_data, base_dt = 'dcz_rqr_dt', label_dt = 'tr_dt', fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        merge_data.loc[merge_data['tr_dt'].isna(), 'tr_dt'] = merge_data.loc[merge_data['tr_dt'].isna(), 'dcq_rqr_dt']
    elif 'total' == div:
        merge_data = rzt_data.merge(dcz_data[dcz_data['now_sts_dsc'] == '5'], on = 'dcz_sqno', how = 'inner')
        merge_data['cnt'] = 1
        merge_data = reindex_base_table(base = merge_data.drop('tr_dt', axis = 1), label = label_data, base_dt = 'dcz_rqr_dt', label_dt = 'tr_dt', fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
        merge_data.loc[merge_data['tr_dt'].isna(), 'tr_dt'] = merge_data.loc[merge_data['tr_dt'].isna(), 'dcq_rqr_dt']
    
    del label_data
    return merge_data

def sus_60d_withdraw_tr(trlog_data, label_data):
    withd_code = ['02']
    expr_list = [f"cptld_tp_kdc in {withd_code}"]
    
    trlog_data = filter_tr_data(trlog_data, expr_list)
    trlog_data['cnt'] = 1
    trlog_data = reindex_base_table(trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    
    del label_data
    return trlog_data

def sus_60d_acno_tr(trlog_data, label_data):
    fc_tr_code = ['01', '02', '04', '05', '08', '09', '12', '13', '14', '15']
    expr_list = [f"cptld_tr_kdc in {fc_tr_code}"]
    
    trlog_data = filter_tr_data(trlog_data, expr_list)
    trlog_data = trlog_data[['cusno', 'tr_dt', 'acno', 'cptld_tr_kdc', 'intg_imps_key_val', 'tram_us']]
    trlog_data['cnt'] = 1
    
    # 거래계좌 정보 중 제외해야 할 정보
    acno_nan_idx           = trlog_data[trlog_data['acno'].isna()].index
    acno_zero_idx          = trlog_data[trlog_data['acno'] == '0'].index
    acno_outward_remit_idx = trlog_data[trlog_data['acno'] == '당발송금'].index
    acno_same_address_idx  = trlog_data[trlog_data['acno'] == '동수취인송금'].index
    acno_exch_idx          = trlog_data[trlog_data['acno'] == '환전거래'].index
    acno_no_address_idx    = trlog_data[trlog_data['acno'] == '수취계좌없음'].index
    acno_inward_remit_idx  = trlog_data[trlog_data['acno'] == '타발송금']
    
    trlog_data = trlog_data.drop(index = acno_nan_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_zero_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_outward_remit_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_same_address_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_exch_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_no_address_idx, axis = 0)
    trlog_data = trlog_data.drop(index = acno_inward_remit_idx, axis = 0)
    trlog_data = handling_acno_type(trlog_data, ['acno'], {'acno' : 'acno_int'})
    trlog_data = reindex_base_table(trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    
    del label_data
    return trlog_data

def dep_tr_intensity(trlog_data, label_data):
    expr_list_dp = [f"cptld_tr_kdc in {['01', '04', '08']}"] # 입금 (01 : 입금, 04 : 영수, 08 : 이체영수)
    expr_list_wd = [f"cptld_tr_kdc in {['02', '05', '09']}"] # 출금 (02 : 출금, 05 : 송금, 09 : 이체송금)
    
    # 입금
    dp_trlog_data = filter_tr_data(trlog_data, expr_list_dp)
    dp_trlog_data = dp_trlog_data[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    dp_trlog_data['cnt'] = 1
    dp_trlog_data = reindex_base_table(dp_trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    dp_trlog_7d_sum_data = add_rolling_feature(dp_trlog_data, 'tram_us', sum, 'tr_dt', '8D', var_names = 'F_7D_Cust_Fc_Dep_Sum')
    
    # 출금
    wd_trlog_data = filter_tr_data(trlog_data, expr_list_wd)
    wd_trlog_data = wd_trlog_data[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    wd_trlog_data['cnt'] = 1
    wd_trlog_data = reindex_base_table(wd_trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    wd_trlog_7d_sum_data = add_rolling_feature(wd_trlog_data, 'tram_us', sum, 'tr_dt', '8D', var_names = 'F_7D_Cust_Fc_Wd_Sum')
    
    # trlog에 입금 / 출금 거래액 merge
    trlog_data = trlog_data.merge(dp_trlog_7d_sum_data[['cusno', 'tr_dt', 'F_7D_Cust_Fc_Dep_Sum']].sort_values(['cusno', 'tr_dt', 'F_7D_Cust_Fc_Dep_Sum']).drop_duplicates(['cusno', 'tr_dt'], keep = 'las'),
                                  on = ['cusno', 'tr_dt'], how = 'left')
    trlog_data = trlog_data.merge(wd_trlog_7d_sum_data[['cusno', 'tr_dt', 'F_7D_Cust_Fc_Wd_Sum']].sort_values(['cusno', 'tr_dt', 'F_7D_Cust_Fc_Wd_Sum']).drop_duplicates(['cusno', 'tr_dt'], keep = 'las'),
                                  on = ['cusno', 'tr_dt'], how = 'left')
    trlog_data = trlog_data[['cusno', 'tr_dt', 'F_7D_Cust_Fc_Dep_Sum', 'F_7D_Cust_Fc_Wd_Sum']].sort_values(['cusno', 'tr_dt', 'F_7D_Cust_Fc_Wd_Sum']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last').reset_index(drop = True)
    
    # 입금 / 출금 누계값 처리
    trlog_data.loc[trlog_data['F_7D_Cust_Fc_Dep_Sum'] == 0, 'F_7D_Cust_Fc_Dep_Sum'] = -1
    trlog_data.loc[trlog_data['F_7D_Cust_Fc_wd_Sum'] == 0, 'F_7D_Cust_Fc_Wd_Sum'] = 0
    
    label_data = append_feature(label_data, trlog_data, fill_value = {'F_7D_Cust_Fc_Dep_Sum' : -1, 'F_7D_Cust_Fc_Wd_Sum' : 0})
    
    # 입금 / 출금 누계값 처리
    label_data.loc[(label_data['F_7D_Cust_Fc_Dep_Sum'] > 0) & (label_data['F_7D_Cust_Fc_Wd_Sum'].isna()), 'F_7D_Cust_Fc_Wd_Sum'] = 0 # 입금 Only (7일간 입금 누계 거래액 > 0, 7일간 출금 누계 거래액 = Na (Na -> 0으로 치환))
    label_data.loc[(label_data['F_7D_Cust_Fc_Dep_Sum'].isna()) & (label_data['F_7D_Cust_Fc_Wd_Sum'] > 0), 'F_7D_Cust_Fc_Dep_Sum'] = -1 # 출금 Only (7일간 입금 누계 거래액 = Na (Na -> -1로 치환), 7일간 출금 누계 거래액 > 0)
    
    # 최근 7일 동안 입금 누계 거래액 대비 출금 누계 거래액 비율
    tmp_data = label_data[['F_7D_Cust_Fc_Dep_Sum', 'F_7D_Cust_Fc_Wd_Sum']]
    tmp_data.loc[tmp_data['F_7D_Cust_Fc_Dep_Sum' == -1, 'F_7D_Cust_Fc_Dep_Sum']] = np.nan
    tmp_data.loc[tmp_data['F_7D_Cust_Fc_Wd_Sum'] == -1, 'F_7D_Cust_Fc_Wd_Sum'] = np.nan
    
    label_data['F_7D_Fc_DepWd_Sum_Ratio'] = (tmp_data['F_7D_Cust_Fc_Wd_Sum'] / tmp_data['F_7D_Cust_Fc_Dep_Sum']) * 100
    del tmp_data
    
    # '입금 강도' 카테고리 컬럼 생성 및 값 입력
    label_data['F_7D_Cust_Fc_Dep_Intensity'] = 0
    label_data.loc[label_data['F_7D_Fc_DepWd_Sum_Ratio'].isna(), 'F_7D_Cust_Fc_Dep_Intensity'] = '1'                                                # 해당 사항 없음 (Ratio = NaN)
    label_data.loc[label_data['F_7D_Fc_DepWd_Sum_Ratio'] == 0, 'F_7D_Cust_Fc_Dep_Intensity'] = '2'                                                  # 입금 Only (Ratio = 0)
    label_data.loc[label_data['F_7D_Fc_DepWd_Sum_Ratio'] < 0, 'F_7D_Cust_Fc_Dep_Intensity'] = '3'                                                   # 출금 Only (Ratio < 0)
    label_data.loc[label_data['F_7D_Fc_DepWd_Sum_Ratio'] >= 0, 'F_7D_Cust_Fc_Dep_Intensity'] = '4'                                                  # 출금 우위 (Ratio >= 0.5)
    label_data.loc[(label_data['F_7D_Fc_DepWd_Sum_Ratio'] > 0) & (label_data['F_7D_Fc_DepWd_Sum_Ratio'] < 0.5), 'F_7D_Cust_Fc_Dep_Intensity'] = '5' # 입금 Only (0 < Ratio < 0.5)
    label_data['F_7D_Cust_Fc_Dep_Intensity'] = label_data['F_7D_Cust_Fc_Dep_Intensity'].astype('category')
    
    label_data = label_data.drop(['F_7D_Cust_Fc_Dep_Sum', 'F_7D_Cust_Fc_Wd_Sum', 'F_7D_Fc_DepWd_Sum_Ratio'], axis = 1)
    return label_data

def service_type(cust, label):
    cust['dptr_acn_cnt'] = cust['dmd_dtpr_accn'] + cust['svtp_dptc_accn'] # 수신
    cust['fx_acn_cnt']   = cust['fx_tr_acn'] # 외환
    cust['cd_acn_cnt']   = cust['cd_tr_acn'] # 카드
    cust['etc_acn_cnt']  = cust['ts_tr_acn'] + cust['gen_la_tr_acn'] + cust['bildc_tr_acn'] + cust['mad_tr_acn'] # 기타
    cust = cust[['cusno', 'dptr_acn_cnt', 'fx_acn_cnt', 'cd_acn_cnt', 'etc_acn_cnt']]
    cust['F_Svc_Type'] = 0
    
    check_dict = {
        (1, 0, 0, 0) : '1', # 수신
        (0, 1, 0, 0) : '2', # 외환
        (0, 0, 1, 0) : '3', # 카드
        (0, 0, 0, 1) : '4', # 기타
        (1, 1, 0, 0) : '5', # 수신 & 외환
        (1, 0, 1, 0) : '6', # 수신 & 카드
        (1, 0, 0, 1) : '7', # 수신 & 기타
        (0, 1, 1, 0) : '8', # 외환 & 카드
        (0, 1, 0, 1) : '9', # 외환 & 기타
        (0, 0, 1, 1) : '10', # 카드 & 기타
        (1, 1, 1, 0) : '11', # 수신 & 외환 & 카드
        (1, 1, 0, 1) : '12', # 수신 & 외환 & 기타
        (1, 0, 1, 1) : '13', # 수신 & 카드 & 기타
        (0, 1, 1, 1) : '14', # 외환 & 카드 & 기타
        (1, 1, 1, 1) : '15', # 수신 & 외환 & 카드 & 기타
        (0, 0, 0, 0) : '16', # 없음 & 없음 & 없음 & 없음
    }
    
    for i in range(len(cust)):
        tmp_dict = [0, 0, 0, 0]
        if 0 < cust['dptr_acn_cnt'][i]:
            tmp_dict[0] = 1
        if 0 < cust['fx_acn_cnt'][i]:
            tmp_dict[1] = 1
        if 0 < cust['cd_acn_cnt'][i]:
            tmp_dict[2] = 1
        if 0 < cust['etc_acn_cnt'][i]:
            tmp_dict[3] = 1
        cust['F_Svc_Type'][i] = check_dict[tuple(tmp_dict)]
    label = append_feature(label, features = cust[['cusno', 'F_Svc_Type']], on_dt = False, dtypes = {'F_Svc_Type' : int})
    label.loc[label['F_Svc_Type'].isna(), 'F_Svc_Type'] = 1
    del cust
    return label

def transaction_time_type(data, label):
    tmp_data = drop_duplicate(data, 'intg_imps_key_val', [19, 20])
    tmp_data['tr_tm'] = pkgs.pd.to_datetime(tmp_data['tr_tm'].fillna('125959'), format = '%H%M%S').dt.time
    
    tmp_data['tr_dt'] = tmp_data['tr_dt'].astype(str)
    tmp_data['tr_tm'] = tmp_data['tr_tm'].astype(str)
    tmp_data['tr_dtm'] = pkgs.pd.to_datetime(tmp_data['tr_dt'] + ' ' + tmp_data['tr_tm'], infer_datetime_format = True)
    tmp_data = tmp_data[['cusno', 'tr_dt', 'tr_tm', 'tr_dtm']]
    
    tmp_m_data = tmp_data.between_time('09:00:00', '12:59:59').reset_index(drop = True) # m : morning (오전)
    tmp_m_data['tr_m_cnt'] = 1
    tmp_m_data['tr_a_cnt'] = 0
    tmp_m_data['tr_w_cnt'] = 0
    
    tmp_a_data = tmp_data.between_time('13:00:00', '15:59:59').reset_index(drop = True) # a : afternoon (오후)
    tmp_a_data['tr_m_cnt'] = 0
    tmp_a_data['tr_a_cnt'] = 1
    tmp_a_data['tr_w_cnt'] = 0
    
    tmp_w_data = tmp_data.between_time('16:00:00', '08:59:59').reset_index(drop = True) # w : outside of work (업무 외)
    tmp_w_data['tr_m_cnt'] = 0
    tmp_w_data['tr_a_cnt'] = 0
    tmp_w_data['tr_w_cnt'] = 1
    
    tmp_m_data= tmp_m_data.drop_duplicates(['cusno', 'tr_dt', 'tr_tm'], keep = 'last').reset_index(drop = True)
    tmp_a_data= tmp_a_data.drop_duplicates(['cusno', 'tr_dt', 'tr_tm'], keep = 'last').reset_index(drop = True)
    tmp_w_data= tmp_w_data.drop_duplicates(['cusno', 'tr_dt', 'tr_tm'], keep = 'last').reset_index(drop = True)
    
    # CUSNO(고객), TR_DT(거래일) 기준 Groupby (시간대 별 거래 건 수 합산)
    tmp_m_sum_data = tmp_m_data.groupby(['cusno', 'tr_dt'])['tr_m_cnt'].agg(sum).reset_index().rename(columns = {'tr_m_cnt' : 'tr_m_sum'})
    tmp_a_sum_data = tmp_a_data.groupby(['cusno', 'tr_dt'])['tr_m_cnt'].agg(sum).reset_index().rename(columns = {'tr_m_cnt' : 'tr_m_sum'})
    tmp_w_sum_data = tmp_w_data.groupby(['cusno', 'tr_dt'])['tr_m_cnt'].agg(sum).reset_index().rename(columns = {'tr_m_cnt' : 'tr_m_sum'})
    
    # 시간대 별 거래 건 수 합산한 df들 merge
    merge_df = tmp_m_sum_data.merge(tmp_a_sum_data[['cusno', 'tr_dt', 'tr_a_sum']].sort_values(['cusno', 'tr_dt', 'tr_a_sum']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last'), on = ['cusno', 'tr_dt'], how = 'outer')
    merge_df = merge_df.merge(tmp_w_sum_data[['cusno', 'tr_dt', 'tmp_w_sum']].sort_values(['cusno', 'tr_dt', 'tr_w_sum']).drop_duplicates(['cusnjo', 'tr_dt'], keep = 'last'), on = ['cusno', 'tr_dt'], how = 'outer')
    merge_df['cusno'] = merge_df['cusno'].astype(str)
    merge_df['tr_dt'] = pkgs.pd.to_datetime(merge_df['tr_dt'], format = '%Y-%m-%d', errors = 'coerce')
    merge_df['tr_m_sum'] = merge_df['tr_m_sum'].fillna(0).astype(int)
    merge_df['tr_a_sum'] = merge_df['tr_a_sum'].fillna(0).astype(int)
    merge_df['tr_w_sum'] = merge_df['tr_w_sum'].fillna(0).astype(int)
    
    # 시간대 별 거래 건 수 Case 별 Category 값 매핑
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_a_sum']) & (merge_df['tr_w_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오전 = 오후, 업무 외 = 0
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_w_sum']) & (merge_df['tr_a_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '2' # 오전 = 업무 외, 오후 = 0
    merge_df.loc[(merge_df['tr_a_sum'] == merge_df['tr_w_sum']) & (merge_df['tr_m_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '2' # 오후 = 업무 외, 오전 = 0
    
    merge_df.loc[(merge_df['tr_m_sum'] > merge_df['tr_a_sum']) & (merge_df['tr_w_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오전 > 오후, 업무 외 = 0
    merge_df.loc[(merge_df['tr_m_sum'] > merge_df['tr_w_sum']) & (merge_df['tr_a_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오전 > 업무 외, 오후 = 0
    
    merge_df.loc[(merge_df['tr_a_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_w_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '1' # 오후 > 오전, 업무 외 = 0
    merge_df.loc[(merge_df['tr_a_sum'] > merge_df['tr_w_sum']) & (merge_df['tr_m_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '1' # 오후 > 업무 외, 오전 = 0
    
    merge_df.loc[(merge_df['tr_w_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_a_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '2' # 업무 외 > 오전, 오후 = 0
    merge_df.loc[(merge_df['tr_w_sum'] > merge_df['tr_a_sum']) & (merge_df['tr_m_sum'] == 0), 'F_1D_Cust_Tr_Time_Type'] = '2' # 업무 외 > 오후, 오전 = 0
    
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_a_sum']) & (merge_df['tr_m_sum'] > merge_df['tr_w_sum']) & (merge_df['tr_a_sum'] > merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오전 = 오후, 오전 > 업무 외, 오후 > 업무 외
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_a_sum']) & (merge_df['tr_w_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_w_sum'] > merge_df['tr_a_sum']), 'F_1D_Cust_Tr_Time_Type'] = '2' # 오전 = 오후, 업무 외 > 오전, 업무 외 > 오후   
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_w_sum']) & (merge_df['tr_a_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_a_sum'] > merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '1' # 오전 = 업무 외, 오후 > 오전, 오후 > 업무 외
    merge_df.loc[(merge_df['tr_a_sum'] == merge_df['tr_w_sum']) & (merge_df['tr_m_sum'] > merge_df['tr_a_sum']) & (merge_df['tr_m_sum'] > merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오후 = 업무 외, 오전 > 오후, 오전 > 업무 외
    merge_df.loc[(merge_df['tr_a_sum'] == merge_df['tr_w_sum']) & (merge_df['tr_a_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_w_sum'] > merge_df['tr_m_sum']), 'F_1D_Cust_Tr_Time_Type'] = '2' # 오후 = 업무 외, 오후 > 오전, 업무 외 > 오전

    merge_df.loc[(merge_df['tr_m_sum'] > merge_df['tr_a_sum']) & (merge_df['tr_m_sum'] > merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '0' # 오전 > 오후, 오전 > 업무 외
    merge_df.loc[(merge_df['tr_a_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_a_sum'] > merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '1' # 오후 > 오전, 오후 > 업무 외
    merge_df.loc[(merge_df['tr_w_sum'] > merge_df['tr_m_sum']) & (merge_df['tr_w_sum'] > merge_df['tr_a_sum']), 'F_1D_Cust_Tr_Time_Type'] = '2' # 업무 외 > 오전, 업무 외 > 오후
    
    merge_df.loc[(merge_df['tr_m_sum'] == merge_df['tr_a_sum']) & (merge_df['tr_m_sum'] == merge_df['tr_w_sum']) * (merge_df['tr_a_sum'] == merge_df['tr_w_sum']), 'F_1D_Cust_Tr_Time_Type'] = '2' # 오전 = 오후 = 업무 외
    
    label = append_feature(label, merge_df[['cusno', 'tr_dt', 'F_1D_Cust_Tr_Time_Type']].sort_values(['cusno', 'tr_dt', 'F_1D_Cust_Tr_Time_Type']).drop_duplicates(['cusno', 'tr_dt'], keep = 'last'), on_dt = True, dtypes = {'F_1D_Cust_Tr_Time_Type' : 'category'})
    label['F_1D_Cust_Tr_Time_Type'] = label['F_1D_Cust_Tr_Time_Type'].astype(str)
    
    if 0 != label[label['F_1D_Cust_Tr_Time_Type'].isna().shape[0]]:
        label.loc[label['F_1D_Cust_Tr_Time_Type'].isna(), 'F_1D_Cust_Tr_Time_Type'] = '3'
        label.loc[label['F_1D_Cust_Tr_Time_Type'] == 'nan', 'F_1D_Cust_Tr_Time_Type'] = '3'
    else:
        pass
    
    del data
    del merge_df
    del tmp_data
    del tmp_m_data
    del tmp_a_data
    del tmp_w_data
    del tmp_m_sum_data
    del tmp_a_sum_data
    del tmp_w_sum_data
    return label

def hrc_fc_tr_ratio(trc, trlog_data, label_data):
    fc_tr_code = ['01', '02', '04', '05', '08', '09', '12', '13', '14', '15']
    
    expr_list1 = [f"fr_bnk_iso_natcd in {trc}"]
    expr_list2 = [f"cptld_tr_kdc in {fc_tr_code}"]
    
    # Total trlog
    total_trlog_data = filter_tr_data(trlog_data, expr_list2)
    total_trlog_data = total_trlog_data[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    total_trlog_data['cnt'] = 1
    total_trlog_data = reindex_base_table(total_trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    
    # Hrc trlog
    hrc_trlog_data = filter_tr_data(trlog_data, expr_list1)
    hrc_trlog_data = filter_tr_data(hrc_trlog_data, expr_list2)
    hrc_trlog_data = hrc_trlog_data[['cusno', 'tr_dt', 'tram_us', 'intg_imps_key_val']]
    hrc_trlog_data['cnt'] = 1
    hrc_trlog_data = reindex_base_table(hrc_trlog_data, label_data, fill = True, fill_column = 'cnt', fill_value = 0, fill_type = int)
    
    del trc
    del fc_tr_code
    del expr_list1
    del expr_list2
    del trlog_data
    del label_data
    return total_trlog_data, hrc_trlog_data

def select_special_account(table, target, select_key):
    new_table = table.copy()
    new_table = new_table[~new_table[target].isna()]
    new_table['tmp_key'] = new_table[target].astype(str)
    new_table['len_key'] = new_table[target].apply(lambda x: len(x))
    new_table['tmp_key'] = new_table['tmp_key'].apply(lambda x: x[:-3])
    
    new_table = new_table[new_table[target].isin(select_key)]
    cusno_list = new_table.cusno.unique()
    return cusno_list

#######################
### MODEL FUNCTIONS ###
#######################
class DATA_PIPELINE:
    def __init__(self, data_dict = None):
        self.data_container = []
        self.total_dataframe = 1
        
        if None == data_dict:
            pass
        else:
            for k in data_dict.keys():
                self.data_container.update({k : data_dict.get(k)})
    
    def data_combine(self):
        if 1 == len(self.data_container.keys()):
            pivot_key = list(self.data_container.keys())[0]
            pivot = self.data_container.get(pivot_key)
            self.total_dataframe = pivot
            return self
        elif 1 < len(self.data_container.keys()):
            pivot_key = list(self.data_container.keys())[0]
            pivot = self.data_container.get(pivot_key)
            self.total_dataframe = pivot
            
            for i in list(self.data_container.keys())[1:]:
                self.total_dataframe = self.total_dataframe.merge(self.data_container.get(i), on = ['cusno', 'tr_dt'], how = 'left')
            return self
        else:
            return self
    
    def drop_columns(self, key_val, list_col):
        data = self.data_container.get(key_val)
        if 'NoneType' == type(data):
            print('Nothing')
        else:
            data = data.drop(list_col, axis = 1)
            self.data_container.update({key_val : data})
        return self
    
    def drop_columns_final(self, list_col):
        data = self.total_dataframe
        if type(1) == type(data):
            print('Nothing')
        else:
            self.total_dataframe = data.drop(list_col, axis = 1)
        return self
    
    def show_type(self, key_val):
        if 'final_df' == key_val:
            if type(1) == type(self.total_dataframe):
                print('Nothing')
            else:
                for i in zip(self.total_dataframe.columns, self.total_dataframe.dtypes):
                    print(i)
        return self
    
    def type_change(self, key_val, col_list, type_name):
        data = self.data_combine.get(key_val)
        
        if 'NoneType' == type(data):
            print('Nothing')
        else:
            if 'int' == type_name:
                data[col_list] = data[col_list].astype(int)
                self.data_container.update({key_val : data})
            elif 'float' == type_name:
                data[col_list] = data[col_list].astype(float)
                self.data_container.update({key_val : data})
            elif 'uint8' == type_name:
                data[col_list] = data[col_list].astype(uint8)
                self.data_container.update({key_val : data})
            elif 'category' == type_name:
                labelencoder = LabelEncoder()
                for col_index in col_list:
                    data[col_index] = labelencoder.fit_transform(data[col_index])
                self.data_container.update({key_val : data})
        return self
    
    def update_table(self, key_val, table):
        data = self.data_container.get(key_val)
        
        if 'NoneType' == type(data):
            print('Nothing')
        else:
            self.data_container.update({key_val : data})
        return self
    
    def add_col(self, key_val, col_name, col):
        data = self.data_container.get(key_val)
        
        if 'NoneType' == type(data):
            print('Nothing')
        else:
            check_list = data.columns
            
            if col_name in check_list:
                print('Already exist')
            else:
                data[col_name] = col
                self.data_container.update({'key_val' : data})
    
    def type_change_final(self, col_list, type_name):
        data = self.total_dataframe
        
        if type(1) == type(data):
            print('Nothing')
        else:
            if 'int' == type_name:
                data[col_list] = data[col_list].astype(int)
                self.total_dataframe = data
            elif 'float' == type_name:
                data[col_list] = data[col_list].astype(float)
                self.total_dataframe = data
            elif 'uint8' == type_name:
                data[col_list] = data[col_list].astype('uint8')
                self.total_dataframe = type_name
            elif 'category' == type_name:
                labelencoder = LabelEncoder()
                for col_index in col_list:
                    data[col_index] = labelencoder.fit_transform(data[col_index])
                self.total_dataframe = data
            else:
                print('No such type')
        return self
    
    def generate_dummy(self, col_list):
        if type(1) == type(self.total_dataframe):
            print('Nothing')
        else:
            self.total_dataframe = pd.get_dummies(self.total_dataframe, columns = col_list)
        return self
    
    def show_df(self):
        if type(1) == type(self.total_dataframe):
            print('Nothing')
        else:
            display(self.total_dataframe)
        return self
    
    def show_sub_df(self, key_val):
        data = self.data_container.get(key_val)
        
        if 'NoneType' == type(data):
            print('Nothing')
        else:
            display(data)
        return self
    
    def add_data_frame(self, dictionary):
        for k in dictionary.keys():
            pre_df = dictionary.get(k)
            pre_df['cusno'] = pre_df['cusno'].astype(str)
            pre_df['tr_dt'] = pd.to_datetime(pre_df['tr_dt'], format = '%Y-%m-%d')
            
            self.data_container.update({k : pre_df})
        return self
    
    def get_total_frame(self):
        if type(1) == type(self.total_dataframe):
            print('Nothing')
        else:
            return self.total_dataframe
    
    def generate_ratio_val(self, col1, col2, col3):
        if type(1) == type(self.total_data_frame):
            print('No model')
        else:
            tmp_one = self.total_dataframe[col1].apply(lambda x: 0 if x < 0 else x)
            tmp_two = self.total_dataframe[col2].apply(lambda x: 0 if x < 0 else x)
            
            final = tmp_one / tmp_two
            final = final.apply(lambda x: -1 if np.isinf(x) else x)
            
            self.total_dataframe[col3] = final
        return self
    
    def generate_delta_val(self, col1, col2, col3):
        if type(1) == type(self.total_dataframe):
            print('No model')
        else:
            tmp_one = self.total_dataframe[col1].apply(lambda x: 0 if x < 0 else x)
            tmp_two = self.total_dataframe[col2].apply(lambda x: 0 if x < 0 else x)
            
            self.total_dataframe[col3] = tmp_one - tmp_two
        return self
    
    def generate_sus_ratio(self, col1, col2, col3):
        if type(1) == type(self.total_dataframe):
            print('No model')
        else:
            final = self.total_dataframe[col1] / self.total_dataframe[col2]
            final = final.apply(lambda x: -1 if np.isnan(x) else x)
            self.total_dataframe[col3] = final
        return self

def xgb_cv(X_tmp, y_tmp, X_val_tmp, y_val_tmp, max_depth, learning_rate, n_esitmators, gamma, min_child_weight, subsample, colsample_bytree, alpha, reg_alpha, reg_lambda):
    model = pkgs.XGBClassifier(
        max_depth        = int(max_depth),
        learning_rate    = learning_rate,
        n_estimators     = int(n_esitmators),
        gamma            = gamma,
        min_child_weight = int(min_child_weight),
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        alpha            = alpha,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        n_jobs = 4, booster = 'gbtree', objective = 'binary:logistic', random_state = 34
    )
    
    model.fit(X_tmp, y_tmp, eval_set = [(X_val_tmp, y_val_tmp)], eval_metric = ['logloss'], early_stopping_rounds = 100, verbose = 100)
    pred_val = model.predict(X_val_tmp)
    f1 = pkgs.f1_score(y_val_tmp, pred_val)
    return f1

def create_label(start_date, end_date, bsn_dsc, cus_tpc, is_train = True):
    # STR - 혐의거래추출결과 테이블
    xtr_rzt = read_table(
        table_name = 'tb_ml_bk_sh_xtr_rzt',
        columns = ['cusno', 'tr_dt', 'dtc_dt', 'sspn_tr_rule_id', 'sspn_tr_stsc'],
        data_types = {'cusno' : str, 'tr_dt' : str, 'dtc_dt' : str, 'sspn_tr_rule_id' : 'category', 'sspn_tr_stsc' : 'category'},
        date_columns = ['tr_dt', 'dtc_dt']
    )
    xtr_rzt = xtr_rzt[(xtr_rzt['dtc_dt'] >= start_date) & (xtr_rzt['dtc_dt'] <= end_date)]
    
    # 공통 - 고객 기본 테이블
    cm_cust = read_table(
        table_name = 'tb_ml_bk_cm_cust',
        columns = ['cusno', 'cus_tpc'],
        data_types = {'cusno' : str, 'cus_tpc' : 'category'}
    )
    
    label = xtr_rzt.merge(cm_cust, on = 'cusno', how = 'left')[['cusno', 'cus_tpc', 'tr_dt', 'dtc_dt', 'sspn_tr_rule_id', 'sspn_tr_stsc']]
    
    rule = pd.read_csv('./rd.csv', encoding = 'cp949', usecols = [0, 3], names = ['sspn_tr_rule_id', 'rule_cls'], header = 0)
    rule = rule[rule['rule_cls'].isin(['수신', '외환'])]
    
    label = label.merge(rule, on = 'sspn_tr_rule_id')
    
    label['bsn_dsc'] = np.where(label['rule_cls'] == '수신', '01', '05')
    label['bsn_dsc'] = label['bsn_dsc'].astype('category')
    label['suspicious'] = np.nan
    
    if is_train:
        label['suspicious'] = np.where(label['sspn_tr_stsc'].isin(['S', 'Z']), 1, 0)
        label['suspicious'] = label['suspicious'].astype('category')
    
    label = label.sort_values(['cusno', 'tr_dt', 'bsn_dsc', 'suspicious']).drop_duplicates(['cusno', 'tr_dt', 'bsn_dsc'], keep = 'last')[['cusno', 'bsn_dsc', 'cus_tpc', 'tr_dt', 'dtc_dt', 'suspicious']]
    label = label[(label['bsn_dsc'] == bsn_dsc) & (label['cus_tpc'].isin([cus_tpc]))].dropna(subset = ['cusno', 'bsn_dsc', 'cus_tpc', 'tr_dt']).reset_index(drop =  True)
    return label

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    #print('오차행렬\n:', confusion)
    #print('정확도 : {:.4f}'.format(accuracy))
    #print('정밀도 : {:.4f}'.format(precision))
    #print('재현율 : {:.4f}'.format(recall))
    #print('F1 : {:.4f}'.format(f1))
    #print('AUC : {:.4f}'.format(auc))    
    return confusion, accuracy, precision, recall, f1, auc

#####################
### ETC FUNCTIONS ###
#####################
def setLogger(name):
    logger = pkgs.logging.getLogger(name) # __name__
    
    formatter = pkgs.logging.Formatter(fmt = '[%(asctime)s | %(levelname)s] : %(message)s', datefmt = '%Y-%m-%d, %H:%M:%S')
    
    fileHandler = pkgs.logging.FileHandler(name, encoding = 'utf-8')
    fileHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.setLevel(level = pkgs.logging.INFO)
    return logger

