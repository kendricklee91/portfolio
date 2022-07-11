# -*- coding: utf-8 -*-
# post process task

import os
import json
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from ksv_model.model_dsr import ModelDSRbase, ModelDSR
from ksv_model.model_mw import ModelMw
import ksv_model.config.const as cst


m_mw = ModelMw()
filter_wip = m_mw.filter_white_ip()

def load_threshold_value(model_type, th_auto):
    # load threshold value json file
    th_file = os.path.join(cst.PATH_CONFIG, 'threshold.json')
    with open(th_file) as json_file:
        th_all = json.load(json_file)
    
    th_md = th_all['th_'+model_type]

    if th_auto:
        return th_md['auto']
    else:
        return th_md['manual']


def update_auto_threshold_value(model_type, df):
    # 1. Get the new threshold values for the model
    new_lower_th_value = df[(df['model_type']==model_type)&(df['action_val']==0)&(df['label_inf']==0)]['prob_inf'].quantile(q=0.25)
    new_upper_th_value = df[(df['model_type']==model_type)&(df['action_val']==1)&(df['label_inf']==1)]['prob_inf'].mean()

    th_file = os.path.join(cst.PATH_CONFIG, 'threshold.json')
    # load threshold value json file
    with open(th_file) as json_file:
        th_all = json.load(json_file)
    
    th_all['th_'+model_type]['auto'][0] = new_lower_th_value
    th_all['th_'+model_type]['auto'][1] = new_upper_th_value

    # save threshold value json file
    with open(th_file, 'w') as json_file:
        json.dump(th_all, json_file, indent=4)


def set_date(date=''):
    dt_today = datetime.datetime.now()
    dt_yesterday = dt_today - datetime.timedelta(days=1)

    if date != '':
        try:
            dt_parsed = parse(date)
            dt_yesterday = dt_parsed
            dt_today = dt_yesterday + datetime.timedelta(days=1)
        except:
            print('Error occured while parsing date. Set to today.')  # error
            return dt_today, dt_yesterday

    return dt_today, dt_yesterday


def task_post_process_supervised(model_type, date='', th_auto=False):
    # Set model type
    allowed_model_type = ['sqli', 'xss', 'rce', 'uaa', 'fup', 'fdn', 'ig']
    if model_type not in allowed_model_type:
        return False

    # Set time period for querying
    dt_today, dt_yesterday = set_date(date)
    
    time_from = dt_yesterday.strftime("%Y%m%d0000")
    time_to = dt_today.strftime("%Y%m%d0000")

    th_val = load_threshold_value(model_type, th_auto)

    if model_type == 'ig':  # Info Gathering model is not payload-based model.
        dsr = ModelDSRbase()
    else:
        dsr = ModelDSR() #(gpu_dev='/gpu:2')

    ### Replaced log query part with reading sample csv file
    df = pd.read_csv(os.path.join(cst.PATH_DATA, 'sqli_pps_1day191105_01.csv'))

    # 1. Recommend dataset to re-train this model
    result, score = dsr.recommend_retrain_dataset(model_type, df, th_val)
    df = result[0]

    # 2. Save payload clustering score
    if score is not None:
        with open(os.path.join(cst.PATH_DATA, model_type+'_pps_clusteringscore_01.json'), 'w') as f:
            json.dump(score, f)
    
    # 3. Update threshold values
    update_auto_threshold_value(model_type, df)

    ### Replaced log save part with writing sample csv file
    df.to_csv(os.path.join(cst.PATH_DATA, 'sqli_pps_1day191105res_01.csv'))



    return df


def task_post_process_ip(date=''):
    # Set time period for querying
    _, dt_yesterday = set_date(date)
    date_y = dt_yesterday.strftime("%Y%m%d")

    ### Replaced db query part with reading sample csv file
    df = pd.read_csv(os.path.join(cst.PATH_DATA, 'post_pps_ipresultsample_01.csv'))

    df['dt_start'] = df['dt_start'].astype(str)
    df['dt_end'] = df['dt_end'].astype(str)
    df['date'] = df['dt_end'].apply(lambda x: parse(x).strftime("%Y%m%d"))
    df = df[df['date']==date_y]

    # 1. IP to query external threat collecting system
    ip_to_query = df[df['ext_threat_q']==1]['ip'].unique().tolist()

    # 2. Create attack/infection IP table by date
    df_pivot = df[['date', 'ip', 'zone', 'model_specific', 'priority']].pivot_table(
                index=['date', 'ip', 'zone'], columns='model_specific',
                values='priority', aggfunc=np.sum, 
                fill_value=0).reset_index().rename_axis(None, axis=1)
    df_blank = pd.DataFrame(columns=['date', 'ip', 'zone', 
                'ig_1', 'dos_1', 'dos_2', 'mw_dns_a', 'mw_dns_i', 'mw_mds_a', 'mw_mds_i', 
                'ext_threat_dt', 'ext_threat_res'])  # add columns to record external threat query result
    df_date = pd.merge(df_pivot, df_blank, how='outer').fillna(0)
    
    # 3. Get high risk IP by sorting
    #  - sorting: date desc > dos_1 desc > dos 2 desc > mw_dns_a desc > mw_dns_i desc > mw_mds_a dessc> ms_mds_i desc > ig_1 desc
    df_ip_risk = df_date.sort_values(['date', 'dos_1', 'dos_2', 
                'mw_dns_a', 'mw_dns_i', 'mw_mds_a', 'mw_mds_i', 'ig_1'],
                ascending=[False, False, False, False, False, False, False, False])

    ### Replaced data save part with writing sample csv file
    df_ip_risk.to_csv(os.path.join(cst.PATH_DATA, 'post_pps_ipresultsampleres_02.csv'))

    return df_ip_risk, ip_to_query


def task_post_process_auto_block(df, model_type, th_auto):
    # 1차 고도화 모델 6개 및 정보수집 모델의 추론 데이터 기반으로 자동 차단 대상 식별

    # 1. 일차 식별 기준은 threshold.json의 모델별 upper value를 활용
    th_val = load_threshold_value(model_type=model_type, th_auto=th_auto)
    df_auto_block = df[(df['model_type']==model_type)&(df['action_val']==0)&(df['label_inf']==1)&(df['prob_inf']>=th_val[1])]

    # 2. 일차 식별된 로그 중 출발지IP가 white list에 포함되어 있는 경우는 필터링하여 자동 차단 대상에서 제외
    mask = filter_wip(df_auto_block['src_ip'])
    df_res = df_auto_block[~mask]

    # 3. 자동 차단 대상(df_res)은 AI 플랫폼에서 적응형 보안관리 시스템으로 REST API 방식으로 전달
    #allocate job here to call REST API
    #df_res['key']

    return df_auto_block


def task_post_process_ig_model_inf(df):
    # 2차 정보수집 모델의 신규 데이터 추론 완료 후, 정보수집 공격하는 출발지IP만 유니크하게 추출하여 날짜와 함께 저장
    # columns: dt_start, dt_end, ip, zone, detect_mode, model_specific, priority, ext_threat_q
    dt_today, dt_yesterday = set_date("")
    date_y = dt_yesterday.strftime("%Y%m%d")
    dt_start = dt_yesterday.strftime("%Y%m%d0000")
    dt_end = dt_today.strftime("%Y%m%d0000")

    df['dt_start'] = dt_start
    df['dt_end']   = dt_end
    df.rename(columns={ 'src_ip' : 'ip'}, inplace=True)
    df['zone'] = 'EXT'
    df['detect_mode'] = ""
    df['model_specific'] = ""
    df['priority'] = ""
    df['ext_threat_q'] = 1

    return df[['dt_start', 'dt_end', 'ip', 'zone', 'detect_mode', 'model_specific', 'priority', 'ext_threat_q']]


def task_post_process_blackip(date=''):
    # 외부위협정보수집시스템으로부터 수집된 IP 정보와 비교하여 기존 저장 정보에 추가 컬럼 기록

    # 1. 오늘자 외부 유해 IP 리스트 뽑기
    # Set time period for querying
    dt_today, dt_yesterday = set_date(date)
    date_y = dt_yesterday.strftime("%Y%m%d")

    ### Replaced log query part with reading sample csv file
    df_blackip = pd.read_csv(os.path.join(cst.PATH_DATA, 'blackip_pps_20191125_01.csv'), encoding='CP949')

    list_blackip_raw = df_blackip['유해IP'].unique().tolist()
    list_blackip = []
    for bips in list_blackip_raw:
        if bips.find(',') > -1:  # multiple black IPs seperated by ','
            list_blackip.extend(bips.split(','))
        else:  # one IP
            list_blackip.append(bips)
    list_blackip = list(set(list_blackip))
    df = pd.DataFrame(list_blackip, columns=['ip'])

    # 2. White IP list 대조하여 내부 IP 제거
    mask = filter_wip(df['ip'])
    df_ex_wip = df[~mask]

    # 3. 수집된 외부위협정보의 IP와 교차 조회
    # DB 형상 미정. 외부위협정보수집시스템을 통해 저장된 위협 IP와 1일 유해IP 비교하여 일치하는 것이 있는지 체크
    ### 위협 Intelligence 저장소의 외부위협 정보에서 일정 기간(정의 필요. 3개월?) IP 읽어온 후, (DB 정의 및 구현 필요)
    # 1일치 유해IP에 mask 적용
    df_res = df_ex_wip#[교집합 mask 적용]

    # 4. 외부위협에서 식별된 IP는 IP 결과에 유해IP 속성으로 저장
    # columns: dt_start, dt_end, ip, zone, detect_mode, model_specific, priority, ext_threat_q
    dt_start = dt_yesterday.strftime("%Y%m%d0000")
    dt_end = dt_today.strftime("%Y%m%d0000")
    
    df_res['dt_start'] = dt_start
    df_res['dt_end'] = dt_end
    df_res['zone'] = 'ext'
    df_res['model_specific'] = 'blackip'
    df_res['priority'] = 3  # 유해IP에 대한 위협 우선순위 값을 정해야 함
    df_res['ext_ghreat_q'] = 0

    # 5. IP 기반 모델 결과 DB에 저장
    ### Replaced data save part with writing sample csv file
    df_res.to_csv(os.path.join(cst.PATH_DATA, 'blackip_pps_20191125res_01.csv'))

    return df_res

def task_post_process_update_ip_info():
    # 분석 대상 IP에 대한 외부위협정보 수집시스템 조회 결과 및 보안 관제 요원의 정밀분석 결과를 반영,
    # 정상으로 식별된 IP를 기록하여 차후 분석 대상 결과에서 제외함
    pass
