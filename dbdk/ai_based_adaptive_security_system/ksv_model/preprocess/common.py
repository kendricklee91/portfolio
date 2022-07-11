# -*- coding: utf-8 -*-
# common preprocess functions

import os
import math
import pandas as pd
import ksv_model.config.const as cst


action_value = {
    'ALLOW' : 0,
    'ALARM' : 0,
    'Alarm' : 0,
    'ALARM AND DEFENCE' : 1,
    'Alarm and Defence' : 1,
    'alarm:0defense:N' : 0, 
    'alarm:0defense:Y' : 1,
    'DEFENCE' : 1,
    'DENY' : 1,
    'PROTECTION' : 1,
    '차단' : 1,
    'DETECTION' : 0,
    'DETECT' : 0,
    '탐지' : 0,
    '차단 + RESET' : 1,
    'BLOCK + RESET' : 1,
    'PASS' : 0,
    'ALIVE' : 0,
    'ACCEPT' : 0,
    'risk:2alarm:1' : 0,
    'etc' : 0
}

def convert_action_value(action_name):
    if type(action_name) != str and math.isnan(action_name):
        action_name = 'etc'
    return action_value[action_name]


def categorize_src_country(src_country):
    if src_country == "Korea, Republic of":
        return 'domestic'
    elif src_country == "Private Network":
        return 'private'
    else:
        return 'overseas'

# 사용법 : test = filter_white_ip(cst)
# DataFrame의 IP 컬럼을 전달 
# rt = test(test_data['src_ip'])
# 해당 row에 대한 값을 Series로 return
def filter_white_ip(df_ip=None):
    # white list ip 목록 
    filter_df = pd.read_csv(os.path.join(cst.PATH_DATA, "mw_ref_whitelist_1.csv"))

    def check_fn(df_ip):
        # white_check 필드는 화이트 목록에 포함된 IP명 1, 아니면 0 
        white_check = df_ip.apply(lambda x : True if x in filter_df['ip'] else False)
        return white_check
    return check_fn
