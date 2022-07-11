# -*- coding: utf-8 -*-
# periodic(batch) process task - DoS, Malware

import os
import json
import pandas as pd
import numpy as np
import datetime, pickle
from dateutil.parser import parse
import ksv_model.config.const as cst
import ksv_model.preprocess.common as cm
from ksv_model.model_mw import ModelMw
from ksv_model.model_dos import ModelDoS


# date  : 기준날짜
# delta : 현재 기준 몇 일 전 날짜인지.... 
def mw_set_date(date=None, delta=None):
    dt_today = datetime.datetime.now()
    dt_yesterday = dt_today - datetime.timedelta(days=1)

    if date != None and delta != None:
        try:
            dt_today = parse(date)
            
            dt_yesterday = dt_today - datetime.timedelta(days=delta)
        except:
            print('Error occured while parsing date. Set to today.')  # error
            
    # 기준 시간만 들어오면 하루 전 리턴 
    elif date != None and delta == None:
        dt_today = parse(date)
        dt_yesterday = dt_today - datetime.timedelta(days=1)

    return dt_today, dt_yesterday


# from_date -> '2020-01-02 04:00'
# to_date   -> '2020-01-02 04:30'
def task_model_dos(from_date, to_date):
    dos = ModelDoS()
    rt = dos.dos_process(from_date, to_date)
    return rt
    
def task_model_mw(type_, today, yesterday):
    mw = ModelMw()

    if type_ == "ip_trend":
        rt = mw.task_ip_trend(today, yesterday)

        return rt 

    elif type_ == "type_1":

        rt = mw.task_model_type_1(today, yesterday)
        
        rt['ps_dt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rt['res_type'] = "type1"

        rt['time_start'] = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        # type1의 경우 일주일간의 DNS의 변화 추이 확인 
        rt['time_end'] = today.strftime("%Y-%m-%d %H:%M:%S")
    
    elif type_ == "type_2":
        rt = mw.task_model_type_2(today, yesterday)
        rt['ps_dt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rt['res_type'] = "type2"
        rt['time_start'] = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        # 배치 실행 되는 날과 그 전날의 외부 IP 차이 
        rt['time_end'] = today.strftime("%Y-%m-%d %H:%M:%S")
    
    elif type_ == "type_3":
        rt = mw.task_model_type_3(today, yesterday)
        rt['ps_dt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rt['res_type'] = "type3"
        rt['time_start'] = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        # 악성코드가 감염된 파일의 기간을 한달로 모니터링
        rt['time_end'] = today.strftime("%Y-%m-%d %H:%M:%S")

    rt['key_ps'] = ""
    rt['ps_id'] = ""
    rt['model_type'] = ""
    rt['inf'] = ""
    rt['dataset_id'] = ""

    return rt