# Denial Of Service


import glob
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator, datetime, pickle, math, re, os, glob
from dateutil.parser import parse


# K-means clustering with standard scaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ksv_model.model_mw import ModelMw
from ksv_model.preprocess.network_analysis import NetworkAnalysis
from ksv_model.preprocess.clustering import KMeansClustering
from ksv_model.preprocess.preprocess_ksv_2nd import KnownAttackPreprocess2
import ksv_model.config.const as cst
import ksv_model.preprocess.common as cm

class ModelDoS:
    def __init__(self):
        self.filter_wip = ModelMw.filter_white_ip()

        with open(os.path.join(cst.PATH_DATA, 'nsims_dict_.pkl'), "rb") as f:
            self.nsims = pickle.load(f)

        # 서비스 거부 공격명 가져오기 
        with open(os.path.join(cst.PATH_DATA, "attack_nm_dos.pkl"), "rb") as f:
            self.attack_names = pickle.load(f)


    def _extract_dstn_ip(self, orig_log):
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
        
    def get_suspicious_ip(self, df_ip):
        dg = self._network_analysis(df_ip)
        
        cd = self._filter_int_ip(nx.out_degree_centrality(dg))
        sorted_cd = sorted(cd.items(), key=operator.itemgetter(1), reverse=True)
        
        df_sorted_cd = pd.DataFrame(sorted_cd, columns=['src_ip', 'cd'])
        
        # 연결정도 중심성이 0.1 이상인 출발지 IP 리스트 뽑기
        # 연결정도 중심성이 0.1 이상인 출발지IP
        df_cd = df_sorted_cd[df_sorted_cd['cd'] > 0.1]
        if df_cd is not None and not df_cd.empty:
            if df_cd.shape[0] > 3:
                res_ddos_cd = df_cd[0:3]['src_ip'].unique().tolist()
            else:
                res_ddos_cd = df_cd['src_ip'].unique().tolist()
        
        wcent = self._filter_int_ip(self._weighted_degree_centrality(dg, True))
        sorted_wcent = sorted(wcent.items(), key=operator.itemgetter(1), reverse=True)
        
        df_sorted_wcd = pd.DataFrame(sorted_wcent, columns=['src_ip', 'wcd'])
                
        df_wcd = df_sorted_wcd[df_sorted_wcd['wcd'] > 0.1]
        
        if df_wcd is not None and not df_wcd.empty:
            if df_wcd.shape[0] > 3:
                res_ddos_wcd = df_wcd[0:3]['src_ip'].unique().tolist()
            else:
                res_ddos_wcd = df_wcd['src_ip'].unique().tolist()
                
        
        ddos_result = list(set(res_ddos_cd + res_ddos_wcd + self._kmeans(df_ip)))

        return ddos_result


    def _filter_arbor_gathering(self, df):
        df_start = df[df['type']=='ALERT_START'][['alarm_id', 'start_time', 'original_log', 'dstn_ip']].copy()
        df_stop = df[df['type']=='ALERT_STOP'][['alarm_id', 'end_time']].copy()

        df_arbor = pd.merge(df_start, df_stop, on='alarm_id')

        df_arbor['start_time'] = df_arbor['start_time'].astype(np.int64)
        df_arbor['start_time'] = df_arbor['start_time'].astype(str)
        df_arbor['end_time'] = df_arbor['end_time'].astype(np.int64)
        df_arbor['end_time'] = df_arbor['end_time'].astype(str)


        df_arbor['ex_dstn_ip'] = df_arbor['original_log'].apply(self._extract_dstn_ip)
        df_arbor = df_arbor[['alarm_id', 'ex_dstn_ip', 'start_time', 'end_time', 'dstn_ip']]
        df_arbor.loc[:, 'start_time'] = df_arbor['start_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M%S'))
        df_arbor.loc[:, 'stop_time'] = df_arbor['end_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M%S'))


        df_arbor = df_arbor.sort_values("stop_time", ascending=False).reset_index()


        print ("df_arbor : ", df_arbor.shape)

        df_arbor['ex_dstn_ip'] = df_arbor[['dstn_ip', 'ex_dstn_ip']].apply(lambda x : x[0] if x[1] == "" else x[1], axis=1)
        df_arbor.sort_values('start_time', inplace=True)

        return df_arbor


    # 1차 Arbor 장비 필터 
    def _filter_1st_arbor_ip(self, df, from_date, to_date):
        # arbor 장비 데이터 
        # 쿼리 : log_code:130404 AND eqp_ip:10.181.142.27 AND alarm_value:"importance 2"
        arbor = pd.read_csv(os.path.join(cst.PATH_DATA, "dos_nsims_arbor_01.csv"), encoding='cp949')
        arbor.rename(columns=self.nsims, inplace=True)

        # DataFrame의 날짜 int형 -> 문자형으로 변환 
        arbor['eqp_dt'] = arbor['eqp_dt'].apply(lambda x: parse(str(x)).strftime("%Y-%m-%d %H:%M")) 
        
        #확인용
        print ("arbor : ", arbor.shape)

        #arbor = arbor.query(" eqp_dt >= '{}' and eqp_dt <= '{}'".format(from_date, to_date))
        arbor = arbor.query(" eqp_dt <= '{}'".format(to_date)).copy()
        arbor_df = self._filter_arbor_gathering(arbor)

        #확인용
        print ("arbor date apply : ", arbor.shape)


        filter_dstn_ip = arbor_df['ex_dstn_ip'].values.tolist()
        # 1차 필터링 
        # 목적지IP 중 ARBOR 장비에서 잡힌 목적지IP는 필터링 대상으로 표시(필터링하여 제외할 대상: 1 / 비대상: 0)
        mask = df["dstn_ip"].apply(lambda x: x in filter_dstn_ip)
        # Arbor 장비에 없는 IP만 추출 
        return df[~mask]


    def _filter_2nd(self, df):
        #  제거해야하는 목록들 
        #  2차 DDoS 장비(130301, 130601)에서 차단된 로그 제거
        mask = ( (df["log_code"] == 130301) | (df["log_code"] == 130601) ) & (df['action_val']==1)
        return df[~mask]
    
    def _filter_3rd(self, df):
        # 3차 필터링 : ‘출발지IP’가 내부 IP, 서버 IP, NAT IP 인 경우 필터링 
        ### -> 보안 사항에 의한 IP 정보 삭제 
        mask = df['src_ip'].apply(lambda x: True if x[0:3]=='' or x[0:7]=='' or x[0:7]=='' or \
                                                    x[0:7]=='' or x[0:8]=='' or x[0:8]=='.' or \
                                                    x[0:4]=='' or x[0:4]=='' else False)
        return df[~mask]                                                    

    def _filter_4th(self, df):
        # 4차 필터링 
        # port 53번 제거, 프로토콜이 ICMP인 데이터 제거 
        df = df[df['src_port'] != 53]
        df = df[df['prtc'] != 'ICMP']

        return df

    def _dos_attack_names(self, df):
        mask = df["attack_nm"].apply(lambda x: True if x in self.attack_names else False)
        return df[mask]

    def dos_process(self, from_date, to_date):
        
        # 서비스 거부 30분 단위 전처리를 위한 클래스  
        dos = KnownAttackPreprocess2()
        
        # 네트워크 분석을 위한 클래스 
        Network = NetworkAnalysis()

        # 클러스터링을 위한 클래스 
        Kmeans = KMeansClustering()

        ######################
        ### --> 대상 데이터  
        # log_code:(140102 150101 150201 150202 130301 130601)
        # 국통망 방화벽 데이터 
        n_fw = pd.read_csv(os.path.join(cst.PATH_DATA, "dos_nsims_n_fw_01.csv"))
        n_fw.rename(columns=self.nsims, inplace=True)
        
        # 날짜 타입형 변환 
        n_fw['recv_time'] = n_fw['recv_time'].apply(lambda x: parse(str(x)).strftime("%Y-%m-%d %H:%M")  )
        n_fw = n_fw.query(" recv_time >= '{}' and recv_time <= '{}'".format(from_date, to_date)).copy()

        n_fw.loc[:, 'n'] = 1
        n_fw.loc[:, 'dj'] = 0
        n_fw.loc[:, 'action_val'] = n_fw['action'].apply(cm.convert_action_value)


        ######################
        ### --> Filter 1 ~ 4차
        ## --> Arbor 1차 차단 
        n_fw_1st = self._filter_1st_arbor_ip(n_fw, from_date, to_date)
        #n_fw_1st.to_csv("./n_fw_1st.csv", index=False)
        
        
        ## --> 2차 DDoS 차단 
        n_fw_2nd = self._filter_2nd(n_fw_1st)


        ## --> filtering white list ip
        mask = self.filter_wip(n_fw_2nd['src_ip'])
        n_fw_2nd = n_fw_2nd[~mask]

        ##--> 3차 필터링 
        n_fw_3rd = self._filter_3rd(n_fw_2nd)

        ## --> 4차 필터링 
        n_fw_4th = self._filter_4th(n_fw_3rd)

        ## --> 최종 DDoS 공격명을 가진 로그만 추출 
        fw_final = self._dos_attack_names(n_fw_4th)
    

        # ######################
        # ### --> Zone 값 EXT 추출 
        # # 국통망 방화벽 로그에서 Zone의 값이 EXT인 IP 데이터만 추출 
        # 쿼리 :log_code:(110101 110102) AND eqp_ip:(대상 ip) 
        # # Step 1 --> 30분 단위 전처리 
        preprocess_1 = dos.preprocess_dos(from_date, to_date)
        
        # # Step 2 --> 네트워크 분석 진행 
        preprocess_nw = Network.get_high_degree_centrality_ip(preprocess_1)

        # # Step 3 --> 클러스터링 진행 
        # # k값 찾기 (최대 10)
        feature = ['count', 'time_diff_sec', 'access_intv', 'access_per_sec', 'unique_src_port_cnt', 'tcp_flag',
                        'duration_std', 'duration_mean', 'unique_dstn_ip_cnt', 'action_val', 'domestic', 'private', 'overseas']
        
        preprocess_1_1 = preprocess_1[feature].copy()
        preprocess_1_1.fillna(0, inplace=True)
        k = Kmeans._find_k_value(10, preprocess_1_1)

        rt_df, score = Kmeans._fit_clustering(k, preprocess_1_1)

        preprocess_1_result = pd.concat([preprocess_1, preprocess_1_1, rt_df], axis=1)
        
        print ("silhouette_coefficient : "  ,  score["silhouette_coefficient"])
        print ("calinski_harabasz_index : " ,  score["calinski_harabasz_index"])
        print ("davies_bouldin_index : "    ,  score["davies_bouldin_index"])


        cluster_list = []
        for index, value in preprocess_1_result['cluster'].value_counts().items():
            if(value < 10):
                cluster_list.append(index)
        
        rt = preprocess_1_result[preprocess_1_result['cluster'].apply(lambda x : x in cluster_list)]

        preprocess_cluster = rt['src_ip'].unique().tolist()
        

        # DDoS 의심 IP AI 플랫폼에 저장
        cluster_rt = pd.DataFrame(data= {'ip' :preprocess_cluster})
        cluster_rt['model_specific'] = 'cluster'
        cluster_rt['detect_model'] = 'dos'

        network_rt = pd.DataFrame(data= {'ip' : preprocess_nw})
        network_rt['model_specific'] = 'network'
        network_rt['detect_model'] = 'dos'

        analysis_rt = pd.concat([cluster_rt, network_rt], axis=0)
        analysis_rt['zone'] = 'EXT'
        analysis_rt['priority'] = 3
        analysis_rt['ext_threat_q'] = "Y"
        analysis_rt['pps_dt'] = ""

        analysis_rt['key_pps'] = ""
        analysis_rt['ps_id']= ""
        analysis_rt['pps_id'] = ""
        analysis_rt['dt_start'] = from_date
        analysis_rt['dt_end'] = to_date
        
        return analysis_rt[["key_pps","ps_id","pps_id","dt_start","dt_end", "ip","zone","detect_model","model_specific","priority","ext_threat_q","pps_dt" ]]