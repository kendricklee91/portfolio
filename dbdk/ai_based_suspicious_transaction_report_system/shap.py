import packages as pkgs
import config as cfg
import utils

logger = utils.set_logger(__name__)

def get_var_map(path):
    var_map = pd.read_csv(path, delimiter = '\t', dtypes = {'old' : float, 'new' : float})
    var_map['description'] = var_map['description'].apply(lambda x: '' if pkgs.pd.isnull(x) else f'[{x}]')
    
    try:
        var_map = ''.join(var_map.split())
        var_map = var_map.applymap(var_map)
    except:
        pass
    return var_map

def shapley_df(model, x_test, result):
    x_test__ = x_test.reset_index(drop = True)
    explainer = pkgs.shap.TreeExplainer(model)
    
    if cfg.SHAPELY_RESULT_SAVE:
        shap_values = explainer.shap_values(x_test__.loc[result.index, :])
    else:
        shap_values = explainer.shap_values(x_test__)
    
    shapely_df = pd.DataFrame(shap_values)
    shapely_df.columns = x_test__.columns
    shapely_df.index = result.index
    return shapely_df

def get_top5_effect_variable(idx, x, df):
    '''
    Predict가 0(비혐의)인 경우 shap value가 음수일수록 비혐의에 영향이 높음
    따라서 변수의 값에 '-' (마이너스)로 변경하고 진행
    '''
    if x[0] == 0:
        x = -x
    
    '''
    TOP5 idx
    * 입력받은 컬럼 중 'predict'를 제외한 나머지 컬럼들의 value 기준 오름차순으로 정렬한 index를 반환 - x[1:].argsort()
    * 출력되는 형태에서 컬럼의 순서는 내림차순으로 반환 (마지막 순서에 있는 컬럼명을 제일 앞순서로) - [::-1]
    '''
    
    top5_idx = x[1:].argsort()[::-1]
    top5_columns = shap_columns[top5_idx][:5]               # shap_columns에서 top5_idx의 전체 index 중 상위 5개 index에 해당하는 변수명 반환
    top5_hangul_columns = shap_hangul_columns[top5_idx][:5] # shap_hangul_columns에서 top5_idx의 전체 index 중 상위 5개 index에 해당하는 한글 변수명 반환
    top5_values = df.loc[idx, top5_columns].values          # df(파생변수 정보가 있는 df)에서 고객의 index, top5_columns에 해당하는 value 반환
    top5_description = shap_description[top5_idx][:5]       # shap_description에서 top5_idx의 전체 index 중 상위 5개 index에 해당하는 descriptiuon 정보 반환
    
    shap_str = map(lambda x, y, z: f"{x}={y:,}{z}" if type(y) != str else f"{x}={y}{z}", top5_hangul_columns, top5_values, top5_description)
    return shap_str

def get_top5_shapely(result, shapely_df, replace_map, var_map, variable_filter, delimiter = '='):
    int_columns = var_map.loc[var_map['type'] == 'int', 'variable'].values
    float_columns = var_map.loc[var_map['type'] == 'float', 'variable'].values
    double_columns = var_map.loc[var_map['type'] == 'double', 'variable'].values
    
    filter_result = result.drop(variable_filter, axis = 1).replace(replace_map)
    
    filter_result.loc[:, int_columns] = filter_result.loc[:, int_columns].astype(int)
    filter_result.loc[:, float_columns] = filter_result.loc[:, float_columns].astype(float)
    filter_result.loc[:, double_columns] = filter_result.loc[:, double_columns].round(0).astype(int)
    
    filter_shapely_df = shapley_df.drop(variable_filter, axis = 1)
    
    result_top5_df = pkgs.pd.concat([filter_result['predict'], filter_shapely_df], axis = 1).apply(lambda x: get_top5_effect_variable(x.name, x, filter_result), result_type = 'expand', axis = 1)
    result_top5_df.columns = ['ai_anss_cntn1', 'ai_anss_cntn2', 'ai_anss_cntn3', 'ai_anss_cntn4', 'ai_anss_cntn5']
    result_top5_df = pkgs.pd.concat([filter_result[['cusno', 'label_dt', 'dtc_dt', 'suspicious', 'predict', '1_proba']], result_top5_df], axis = 1).rename(columns = {'1_proba' : 'aisnss_str', 'label_dt' : 'tr_dt'})
    return result_top5_df
    
def shapely_top5_result(model, x_test, result):
    logger.info('Model의 shapely 결과 산출')
    
    var_map = get_var_map(cfg.NH_AML_VARIABLE_DEF_FILE_PATH)
    var_filter = pd.read_csv(cfg.NM_AML_VARIABLE_FILTER_FILE_PATH, DELIMITER = '\t')['variable'].values

    # filter 적용할 feature select
    var_map = var_map[(~var_map['variable'].isin(var_filter)) & (var_map['cus_tpc'].isin(['공통', '개인']))]
    tmp_replace=  var_map[~var_map['old'].isna()][['variable', 'old', 'new']].set_index('variable').to_dict('split')

    replace_dict = {}
    for k, v in zip(tmp_replace['index'], tmp_replace['date']):
        replace_dict[k] = {v[0] : v[1]}
        
    shapely_df = shapely_df(model, x_test, result)
    shapely_cols = shapely_df.columns.drop(var_filter)
    shapely_hangul_cols = shapely_cols.map(var_map.set_index('variable')['hangul_variable'].to_dict())
    shapely_description = shapely_cols.map(var_map.set_index('variable')['description'].to_dict())
    
    shapely_top5_result = get_top5_shapely(result, shapely_df, replace_dict, var_map, var_filter)
    return shapely_top5_result