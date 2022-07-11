from __future__ import print_function

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import itertools
import datetime, re, pickle
import time, json, csv, os, sys, pickle
import ksv_model.preprocess.common as cm
import ksv_model.config.const as cst
from dateutil.parser import parse

tcp_flag = ['3Way OK / FIN1 [SAF:SA]', '3Way OK / FIN2 [SAF:SAF]']
risk_dict = {
        np.nan   : 0,
        '2'      : 2,
        'Low'    : 1,
        'Middle' : 2,
        'High'   : 3 
    }

with open(os.path.join(cst.PATH_DATA, "nsims_dict_.pkl"), "rb") as f:
    nsims = pickle.load(f)
    
class KnownAttackPreprocess2:
    
    def __init__(self):
        # 필터 대상  장비 리스트 
        with open(os.path.join(cst.PATH_DATA, "ig_ref_eqpipfilter.pkl"), "rb") as f:
            self.eqp_ip = pickle.load(f)        
        
        # 공격명 목록 가져오기 
        with open(os.path.join(cst.PATH_DATA, "ig_ref_attacknm.pkl"), "rb") as f:
            self.vul_attack_names = pickle.load(f)        
    
        # 서비스 거부 공격명 가져오기 
        with open(os.path.join(cst.PATH_DATA, "attack_nm_dos.pkl"), "rb") as f:
            self.attack_names = pickle.load(f)


    def _ip_trans(self, ip):
        # 정상적인 ip인지 확인 
        if(ip.count(".") != 3):
            return 0
        else:
            a, b, c, d = ip.split(".")
            return [int(a), int(b), int(c), int(d)]

    def _get_data_df(self, load_file_dir):
        try:
            df = pd.read_csv(load_file_dir, encoding='cp949')
        except:
            df = pd.read_csv(load_file_dir, encoding='utf-8')

        # 컬러명 한글에서 영어로... 파일 하드코딩 
        with open(os.path.join(cst.PATH_DATA,"nsims_dict_.pkl"), "rb") as f:
            nsims = pickle.load(f)

        df.rename(columns=nsims,inplace=True) 
        return df

    def _get_data_label_df(self, load_file_dir):
        try:
            df = pd.read_csv(load_file_dir, encoding='cp949')
        except:
            df = pd.read_csv(load_file_dir, encoding='utf-8')

        return df

    def get_save_df(self, save_df, save_file_dir):
        if save_df.empty or save_df is None:
            print("Data None") # Error log level
            
            return False
        else:
            save_df.to_csv(save_file_dir, index=False)
            
            return True

    # df 파일은 IDS, IPS 데이터 수집 데이터
    # df    : 전처리 DataFrame
    # type_ : 학습 및 추론 선택 
    # df_y  : 라벨링 데이터
    # tdiff : 통계 데이터 수집 시간 
    def ig_preprocess(self, df, df_y=None,  tdiff=120, type_="train" ):
        print (df.columns)
        ###################################################
        ###################################################

        # eqp_ip 필터링 
        df['filter'] = df['eqp_ip'].apply(lambda x : x in self.eqp_ip)
        df = df[df['filter']].copy()
        
        df = df.sort_values("recv_time", ascending=True).reset_index()
        
        # 공격명이 있는 것만 
        df["vul_attack"] = df["attack_nm"].apply(lambda x: 1 if x in self.vul_attack_names else 0)
        
        # 공격 유형이 있는 데이터만 추출 
        df_v = df[df["vul_attack"]==1].copy()
        df_v = df_v.sort_values("recv_time", ascending=True).reset_index()
        
        # CSV에서 읽어오는 데이터에서는 tdiff 이하의 데이터가 없기 때문에 
        # 00:02:00 이상 데이터만 가져온다. (하드코딩)
        #df_v = df_v[df_v['recv_time'] >= (int(str(df_v['recv_time'].iloc[0])[:8] + "000000") + 200)].copy()
        tdiff_time = datetime.datetime.strptime(str(df_v['recv_time'].iloc[0]), '%Y%m%d%H%M%S') + datetime.timedelta(minutes = (tdiff/60))
        df_v = df_v[df_v['recv_time'] >= int(tdiff_time.strftime('%Y%m%d%H%M%S'))].copy()

        # df_v의 하나의 row값이 학습 및 추론에 사용될 데이터임 
        df_v['count'] = df["count"] = 1
        df_v['log_time']    = df_v['recv_time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S'))
        df_v['action_val']  = df_v['action'].apply(cm.convert_action_value)
        df_v["src_country"] = df_v['src_country_name'].apply(cm.categorize_src_country)
        df_v["domestic"]    = df_v["src_country"].apply(lambda x: 1 if x == 'domestic' else 0)
        df_v["private"]     = df_v["src_country"].apply(lambda x: 1 if x == 'private' else 0)
        df_v["overseas"]    = df_v["src_country"].apply(lambda x: 1 if x == 'overseas' else 0)
        
        # 집계할 때 필요함
        df['action_val']  = df['action'].apply(cm.convert_action_value)
        
        # 통계에 사용할 데이터의 시간 데이터로 변환 

        df['log_time'] = df['recv_time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S'))


        # df_vc 데이터에 파생 변수를 붙인다. 
        df_vc = df_v.copy()
        
        # df_v의 각 row값에 5분간 이벤트 로그의 통계값을 붙인다. 
        print(df_v.shape)
        for idx, row in df_v.iterrows():
            if(idx % 1000 == 0):
                print(idx)
            # 현재 index 레코드의 출발지 IP
            src_ip = row['src_ip']
            # 현재 index 레코드에 대한 직전 5분 temporary dataframe
            time_end = row['log_time']
            time_start = time_end - datetime.timedelta(0, tdiff)  # (days, seconds)
            
            ####################################################################
            ####################################################################
            # 통계량 파생변수를 만들 데이터셋 DataFrame를 생성 
            # df_v의 출발지IP 기준 time_start ~ time_end까지 데이터 
            # 플랫폼에선 쿼리로 대체할지 확인해야 함 
            df_tmp = df[(df['log_time'] >= time_start) & (df['log_time'] <= time_end) & \
                        (df['src_ip']==src_ip)].copy().sort_values("log_time", ascending=False)
            ####################################################################
            ####################################################################
            
            # count
            df_vc.loc[idx, "count"] = df_tmp["count"].sum()
            # 최초 시간, 최후 시간의 시간 차(timedelta)
            df_vc.loc[idx, "time_diff"] = df_tmp["log_time"].max() - df_tmp["log_time"].min()
            # action_val
            df_vc.loc[idx, "action_val"] = df_tmp["action_val"].mean()
            # unique_attack_cnt
            df_vc.loc[idx, "unique_attack_cnt"] = len(df_tmp["attack_nm"].unique())
            # unique_src_port_cnt
            df_vc.loc[idx, "unique_src_port_cnt"] = len(df_tmp['src_port'].unique())
            # unique_dstn_ip_cnt
            df_vc.loc[idx, "unique_dstn_ip_cnt"] = len(df_tmp["dstn_ip"].unique())
            # packet_size
            df_vc.loc[idx, "packet_size_mean"] = df_tmp["pkt_size"].mean()
            # vul_attack
            df_vc.loc[idx, "vul_attack"] = df_tmp["vul_attack"].mean()
            # diff var.
            df_vc.loc[idx, "diff_std"] = df_tmp['log_time'].diff().std()
            
        # loop 처리 후 일괄적인 파생변수 처리
        df_vc['time_diff_sec'] = df_vc['time_diff'].apply(lambda x: x.seconds)
        df_vc['attack_intv'] = df_vc['time_diff_sec'] / df_vc['count']
        # 1회 공격일 경우 tdiff초 입력
        df_vc.loc[df_vc['time_diff_sec'] == 0, 'attack_intv'] = tdiff  
        df_vc['attack_per_sec'] = df_vc['count'] / df_vc['time_diff_sec']
        # 1회 공격일 경우 tdiff초 입력
        df_vc.loc[df_vc['time_diff_sec'] == 0, 'attack_per_sec'] = 1/tdiff  
    
        # src_ip 분할 추가 
        ip_tmp = df_vc['src_ip'].apply(self._ip_trans)
        ip_result = pd.DataFrame(ip_tmp.values.tolist(), columns=['src_ip_a', 'src_ip_b', 'src_ip_c', 'src_ip_d'])

        ip_tmp2 = df_vc['dstn_ip'].apply(self._ip_trans)
        ip_result2 = pd.DataFrame(ip_tmp2.values.tolist(), columns=['dstn_ip_a', 'dstn_ip_b', 'dstn_ip_c', 'dstn_ip_d'])
        
        df_vc.reset_index(drop=True, inplace=True)
        df_vc_y = pd.concat([df_vc, ip_result, ip_result2], axis=1)
        
        #df_vc_y['src_ip']
        #df_vc_y['dstn_ip']

        # timedelta를 int형으로 변환 
        df_vc_y['diff_std'] = df_vc_y['diff_std'].apply(lambda x : x.seconds)
        df_vc_y['time_diff'] = df_vc_y['time_diff'].apply(lambda x : x.seconds)
            
        # risk에 대한 값 처리 
        df_vc_y['risk'] = df_vc_y['risk'].apply(lambda x : risk_dict[x])
        
        # array(['TCP', 'UDP', 'ICMP'], dtype=object)
        df_vc_y["prtc_udp"] = df_vc_y["prtc"].apply(lambda x: 1 if x == 'UDP' else 0)
        df_vc_y["prtc_tcp"] = df_vc_y["prtc"].apply(lambda x: 1 if x == 'TCP' else 0)
        df_vc_y["prtc_ip"] = df_vc_y["prtc"].apply(lambda x: 1 if x == 'IP' else 0)
        df_vc_y["prtc_icmp"] = df_vc_y["prtc"].apply(lambda x: 1 if x == 'ICMP' else 0)
        

        # {'src_port', 'dstn_port', 'pkt_cnt', 'pkt_size', 'risk', }
            # 학습 및 추론에 사용되는 컬럼 추출(32 Col)
        extract_columns = ["src_ip_a", "src_ip_b", "src_ip_c", "src_ip_d", "src_port", 'dstn_port', 'risk', 
                           'dstn_ip_a', 'dstn_ip_b', 'dstn_ip_c', 'dstn_ip_d',
                           'pkt_cnt', 'pkt_size', "vul_attack","count", "action_val","domestic","private",
                           'time_diff', 'diff_std', 'prtc_udp', 'prtc_tcp', 'prtc_ip', 'prtc_icmp',
                           "overseas","unique_attack_cnt","unique_src_port_cnt","unique_dstn_ip_cnt",
                           "packet_size_mean", "time_diff_sec","attack_intv","attack_per_sec"]

        # nsims key 및 전처리 부수 컬럼들 추가 
        extract_col_return = ['src_ip', 'key_ps', 'key', 'ps_id', 'model_type', 'inf', 'dataset_id', 'ps_dt', 
                              "src_ip_a", "src_ip_b", "src_ip_c", "src_ip_d", "src_port", 'dstn_port', 'risk', 
                              'dstn_ip_a', 'dstn_ip_b', 'dstn_ip_c', 'dstn_ip_d',
                              'pkt_cnt', 'pkt_size', "vul_attack","count", "action_val","domestic","private",
                              'time_diff', 'diff_std', 'prtc_udp', 'prtc_tcp', 'prtc_ip', 'prtc_icmp',
                              "overseas","unique_attack_cnt","unique_src_port_cnt","unique_dstn_ip_cnt",
                              "packet_size_mean", "time_diff_sec","attack_intv","attack_per_sec", 'label']

        if(type_=="train"):
            # label 데이터 병합 
            df_vc = pd.merge(df_vc_y, df_y, on="src_ip") 

            dataset = df_vc[extract_columns].copy()

            
            # 결측치는 0으로 치환한다. 
            dataset.fillna(0, inplace=True)
        
            ### --> min_max 적용
            # MinMaxScaler에서 Y도 변화하는 것을 방지하기 위해서 따로 저장
            x = dataset.values
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,255))

            x_scaled = min_max_scaler.fit_transform(x)
            df_scaler = pd.DataFrame(x_scaled)
            df_scaler.columns = dataset.columns
            dataset = df_scaler.copy()

            # label 값 붙이기 
            dataset = pd.concat([dataset, df_vc[['src_ip', 'key', 'label']]], axis=1)
            # 전처리 적용 로그 key
            dataset['key_ps'] = ""
            # 적용 전처리기 ID
            dataset['ps_id'] = "ig"
            dataset['model_type'] = "ig"
            dataset['dataset_id'] = 'ig'
            dataset['inf'] = 0
            dataset['ps_dt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return dataset[extract_col_return]

        elif(type_ == "inference"):
            dataset = df_vc_y[extract_columns].copy()
            # 결측치는 0으로 치환한다. 
            dataset.fillna(0, inplace=True)
        
            ### --> min_max 적용
            x = dataset.values
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,255))

            x_scaled = min_max_scaler.fit_transform(x)
            df_scaler = pd.DataFrame(x_scaled)
            df_scaler.columns = dataset.columns
            dataset = df_scaler.copy()

            dataset = pd.concat([dataset, df_vc[['src_ip', 'key']]], axis=1)
            # 전처리 적용 로그 key
            dataset['key_ps'] = ""
            dataset['label'] = ""
            # 적용 전처리기 ID
            dataset['ps_id'] = "ig"
            dataset['model_type'] = "ig"
            dataset['dataset_id'] = 'ig'
            dataset['inf'] = 1
            dataset['ps_dt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return dataset[extract_col_return]
        

    def extract_dstn_ip(self, orig_log):
        p_mobj = re.compile(r"managed.object")
        p_ip = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

        for s in orig_log.split(', '):
            if p_mobj.search(s):
                dstn_str = s
                break
        
        ip_lst = p_ip.findall(dstn_str)
        if len(ip_lst) > 0:
            return ip_lst[0]
        else:
            return ""






    def preprocess_dos(self, from_date, to_date,  tdiff=1800):

        # ######################
        # ### --> 실제 대상 로그 
        # # 대전센터망 방화벽 로그
        # 쿼리 : log_code:(110401 110402) AND eqp_ip:(10.180.195.123 10.180.195.124 10.180.33.31 10.180.33.32) AND ingres_if:(eth10 up1011) AND src_ip:( 대상 iP)
        df_dj_fw = pd.read_csv(os.path.join(cst.PATH_DATA, "dos_nsims_dj_fw_01.csv"), encoding='cp949')
        df_dj_fw.rename(columns=nsims, inplace=True)
        
        df_dj_fw['recv_time'] = df_dj_fw['recv_time'].apply(lambda x: parse(str(x)).strftime("%Y-%m-%d %H:%M:%S") )

        print ("df_dj : ", df_dj_fw.shape)
        df_dj_fw = df_dj_fw.query(" recv_time >= '{}' and recv_time <= '{}' ".format(from_date, to_date)).copy()
        print ("date df_dj : ", df_dj_fw.shape)
        


        df_dj_fw.loc[:, "count"]       = 1
        df_dj_fw.loc[:, 'action_val']  = df_dj_fw['action'].apply(cm.convert_action_value)
        df_dj_fw.loc[:, "src_country"] = df_dj_fw["src_country_name"].apply(cm.categorize_src_country)
        df_dj_fw.loc[:, "domestic"]    = df_dj_fw["src_country"].apply(lambda x: 1 if x == 'domestic' else 0)
        df_dj_fw.loc[:, "private"]     = df_dj_fw["src_country"].apply(lambda x: 1 if x == 'private' else 0)
        df_dj_fw.loc[:, "overseas"]    = df_dj_fw["src_country"].apply(lambda x: 1 if x == 'overseas' else 0)
        #df_dj_fw['recv_time']           = df_dj_fw['recv_time'].astype(str)
        df_dj_fw.loc[:, 'log_time']    = df_dj_fw['recv_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_dj_fw['tcp_flag']           = df_dj_fw['tcp_flag'].apply(lambda x : 1 if x in tcp_flag else 0 )
        
        # 대전 방화벽 로그 복사 
        df_tmp = df_dj_fw.copy()
        # src_ip, dstn_ip 기준으로 sum
        df_ip = df_tmp[["src_ip", "dstn_ip", "count"]].groupby(["src_ip", "dstn_ip"]).sum().reset_index()

        # 기간 내 로그 시작 시간과 끝 시간을 구해서 시간 차 구함
        df_ip_time_min = df_tmp[["src_ip", "dstn_ip", "log_time"]].groupby(["src_ip", "dstn_ip"]).min().reset_index()
        df_ip_time_min.rename(columns={'log_time':"time_start"}, inplace=True)

        df_ip_time_max = df_tmp[["src_ip", "dstn_ip", "log_time"]].groupby(["src_ip", "dstn_ip"]).max().reset_index()
        df_ip_time_max.rename(columns={'log_time':"time_end"}, inplace=True)

        df_ip_time = pd.merge(df_ip_time_min, df_ip_time_max, on=["src_ip", "dstn_ip"])
        df_ip_time.loc[:, 'time_diff'] = df_ip_time["time_end"] - df_ip_time["time_start"]
        df_ip = pd.merge(df_ip, df_ip_time[["src_ip", "dstn_ip", 'time_start', 'time_end', 'time_diff']], on=["src_ip", "dstn_ip"])

        # --> 추가
        # tcp flag 정보 
        df_tcp = df_tmp[["src_ip", "dstn_ip", "tcp_flag"]].groupby(['src_ip', 'dstn_ip']).sum().reset_index()
        df_ip = pd.merge(df_ip, df_tcp, on=['src_ip', 'dstn_ip'])

        # --> 추가 
        # 유지시간 
        df_duration = df_tmp[["src_ip", "dstn_ip", "duration"]].groupby(['src_ip', 'dstn_ip']).std().reset_index()    
        df_duration.rename(columns={'duration':'duration_std'}, inplace=True)
        df_ip = pd.merge(df_ip, df_duration, on=['src_ip', 'dstn_ip'])

        df_duration = df_tmp[["src_ip", "dstn_ip", "duration"]].groupby(['src_ip', 'dstn_ip']).mean().reset_index()    
        df_duration.rename(columns={'duration': 'duration_mean'}, inplace=True)
        df_ip = pd.merge(df_ip, df_duration, on=['src_ip', 'dstn_ip'])

        # 시간 차를 second로 저장
        df_ip.loc[:, 'time_diff_sec'] = df_ip['time_diff'].apply(lambda x: x.seconds)
        # 액세스 로그당 시간(초)
        df_ip.loc[:, 'access_intv'] = df_ip['time_diff_sec'] / df_ip['count']
        df_ip.loc[df_ip['time_diff_sec'] == 0, 'access_intv'] = tdiff  # 1회 공격일 경우 tdiff초 입력
        # 시간(초)당 액세스
        df_ip.loc[:, 'access_per_sec'] = df_ip['count'] / df_ip['time_diff_sec']
        df_ip.loc[df_ip['time_diff_sec'] == 0, 'access_per_sec'] = 1/tdiff  # 1회 공격일 경우 tdiff초 입력

        #src_ip-dstn_ip에서 사용한 unique한 src port 수
        df_src_port = df_tmp[["src_ip", "dstn_ip", "src_port", "count"]].groupby(["src_ip", "dstn_ip", "src_port"]).sum().reset_index()
        df_src_port['unique_src_port_cnt'] = 1
        df_src_port2 = df_src_port[["src_ip", "dstn_ip", "unique_src_port_cnt"]].groupby(["src_ip", "dstn_ip"]).sum().reset_index().sort_values("unique_src_port_cnt", ascending=False)
        df_ip = pd.merge(df_ip, df_src_port2, on=["src_ip", "dstn_ip"])

        df_dstn_ip = df_tmp[["src_ip", "dstn_ip", "count"]].groupby(["src_ip", "dstn_ip"]).sum().reset_index()
        df_dstn_ip['unique_dstn_ip_cnt'] = 1
        df_dstn_ip2 = df_dstn_ip[["src_ip", "unique_dstn_ip_cnt"]].groupby("src_ip").sum().reset_index().sort_values("unique_dstn_ip_cnt", ascending=False)
        df_ip = pd.merge(df_ip, df_dstn_ip2, on="src_ip")

        df_action = df_tmp[["src_ip", "action_val"]].groupby("src_ip").mean().reset_index()
        df_ip = pd.merge(df_ip, df_action, on="src_ip")
        df_ip = pd.merge(df_ip, df_tmp[["src_ip", "src_country", "domestic", "private", "overseas"]].groupby("src_ip").min(), on="src_ip")
        
        return df_ip