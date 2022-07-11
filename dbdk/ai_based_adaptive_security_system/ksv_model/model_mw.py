# malware
import os, pickle, re
import pandas as pd
import datetime
from dateutil.parser import parse

import ksv_model.config.const as cst
# from ksv_model.periodic_process import mw_set_date

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

    
class ModelMw():
    def __init__(self):
        # 현재 가지고 있는 csv 파일의 한글 컬럼명을 영문으로 변환 
        with open(os.path.join(cst.PATH_DATA, "nsims_dict_.pkl"), "rb") as f:
            self.nsims_dict = pickle.load(f)
        
        self.filter_white = ModelMw.filter_white_ip()


    @staticmethod
    # 사용법 : test = filter_white_ip()
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
    

    ### Seculayer의 트렌드 분석이라는 기능을 사용 
    ### 현재는 Seculayer의 트렌드 분석을 사용할 수 없기 때문에 
    ### 데이터가 파이썬에서 나온 df처럼 나온다고 가정 
    # trend_from   --> 2019-12-22
    # trend_to     --> 2019-12-15
    def type_1_trend_analysis(self, from_, to_):
        ### 트렌드 분석 결과 
        ### from ~ to  기간 데이터 
        ### 배치날짜 ~ -7일 
        trend = pd.read_csv(os.path.join(cst.PATH_DATA,"mw_ps_type1_iptrend.csv"), parse_dates=[1])
        trend = trend.query(" date <= '{}' and date >= '{}' ".format(from_.strftime("%Y-%m-%d"), to_.strftime("%Y-%m-%d")))
        trend = trend.pivot_table(index=['src_ip'], columns='date', values='cnt').reset_index()

        ### 현재 내부 DNS와 알려진 DNS주소는 백차장님이 작성하고 있고, 
        # 보고서에 사용된 데이터를 기반으로 
        
        ###############################################
        # filter에 있는 데이터 제거 
        #################  f i l t e r  ###############
        mask = self.filter_white(trend['src_ip'])
        result = trend[~mask]
        ############################################### 
        
        # 해당 IP 기준으로 대전 방화벽에서 행위 분석 진행 
        check_ip = result['src_ip'].tolist()
        ## ---> 대전 센터망 
        # 위의 check_ip ip 리스트에서 대전 방화벽 데이터 추출 
        # log_code:(110101) AND ingres_if:INT AND dstn_port:53 AND src_ip:(check_ip 입력)""" 
        check_ip_dw_data = pd.read_csv(os.path.join(cst.PATH_DATA, "mw_nsims_type1_dw_fw.csv"))
        check_ip_dw_data.rename(columns = self.nsims_dict, inplace = True)

        # 배치 실행 하고 하루치 데이터를 가져온다. 
        # 배치가 2019-12-22 00:00에 돌고 실제 타켓 날짜는 2019-12-21
        # taget_day에 2019-12-21일 날짜를 가져온다. 
        _, target_day = mw_set_date(str(from_))

        check_ip_dw_data['date'] = check_ip_dw_data['gather_time'].apply(lambda x : parse(str(x)).strftime("%Y%m%d"))
        check_ip_dw_data = check_ip_dw_data[check_ip_dw_data['date'] == target_day.strftime("%Y%m%d")]

        ## --> 국통 센터망 
        # 위의 check_ip ip 리스트에서 국통망 방화벽 데이터 추출 
        # log_code:(110101) AND ingres_if:INT AND dstn_port:53 AND src_ip:(check_ip 입력)""" 
        check_ip_n_data = pd.read_csv(os.path.join(cst.PATH_DATA, "mw_nsims_type1_n_fw.csv"))
        check_ip_n_data.rename(columns = self.nsims_dict, inplace = True)


        # 배치 실행 하고 하루치 데이터를 가져온다. 
        # 배치가 2019-12-22 00:00에 돌고 실제 타켓 날짜는 2019-12-21 
        check_ip_n_data['date'] = check_ip_n_data['gather_time'].apply(lambda x : parse(str(x)).strftime("%Y%m%d"))
        check_ip_n_data = check_ip_n_data[check_ip_n_data['date'] == target_day.strftime("%Y%m%d")]

        common_dstn_ip = pd.Series(list(set(check_ip_dw_data['dstn_ip']) & set(check_ip_n_data['dstn_ip'])))

        # White List Ip 제거
        rt = common_dstn_ip[~self.filter_white(common_dstn_ip)]
        
        return rt


    # 아래 함수는 해당 IP의 하루마다 접속한 외부 IP를 추출하여 
    # 전날 대비 신규로 접속한 외부 IP를 추출한다. 
    def type_2_query_diff(self, ip, date_from, date_to):
        # 플랫폼에서는 아래 쿼리의 결과를가져온다. 
        # seudo code에서는 csv에서 파일을 받아서 사용한다. 아래 쿼리는 src_ip가 대상 IP라고 가정 
        #query = """log_code:110401 eqp_ip:(장비 IP) AND src_ip:대상IP AND ingres_if:(eth11 down2223)"""
        # 해당 IP의 현재날짜, 현재날짜 - 1일 위의 쿼리를 사용하여 방화벽 로그를 가져온다. 
        ##########################################
        # pseudo code에서는 파일로 대체한다. 
        # 플랫폼에서 구현이 될 때는 해당 csv파일은 쿼리 혹은 db로 처리되어야 한다. 
        data = pd.read_csv(os.path.join(cst.PATH_DATA, "mw_nsims_type2_20191220~21.csv"))
        data['date'] = data['gather_time'].apply(lambda x : str(x)[:8])
        
        tmp_20191221 = data.query(" date == '{}' ".format(date_from.strftime("%Y%m%d")))    
        tmp_20191220 = data.query(" date == '{}' ".format(date_to.strftime("%Y%m%d")))    
        
        # 2019-12-29일 목적지 IP 비교하여 2019-12-30일에 신규 IP 출력 
        rt =  list(set(tmp_20191220['dstn_ip'].unique()) - set(tmp_20191221['dstn_ip'].unique()))
        #########################################
        
        return rt
    
    def type_3_query_dstn_ip(self, ip_df, from_date, to_date):
        # 악성코드에 감염되었다고 의심되는 IP가 해당 이벤트 로그 발생 시점부터 
        # 아래 쿼리 결과(1일치)
        # 대상 시스템 : 국통망 
        # 쿼리 : (log_code:110101 AND src_ip:(대상IP )) AND eqp_ip:장비IP
        # 쿼리 검색 기간 : gather_time(수집시간) + 3일 데이터 수집 
        # 데이터 목적 : 해당 IP의 3일 동안 발생된 외부 dstn_ip 추출 
        type_3_dstn_ip = pd.read_csv(os.path.join(cst.PATH_DATA,"mw_nsims_type3_dstn_ip.csv"))
        type_3_dstn_ip.rename(columns=self.nsims_dict, inplace=True)

        # type_3_dstn_ip['gather_time'] = type_3_dstn_ip['gather_time'].astype(str)
        # type_3_dstn_ip['date'] = type_3_dstn_ip['gather_time'].apply(lambda x: parse(x).strftime("%Y%m%d"))
        # type_3_dstn_ip = type_3_dstn_ip.query(" date < '{}' and  date >= '{}' ".format(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d")))
    


        ip_target = []
        ip_list = []
        for ip in ip_df['src_ip'].tolist():
            tmp = type_3_dstn_ip.query(" src_ip == '{}'".format(ip))
            tmp_rt = tmp['dstn_ip'].unique().tolist()
            ip_target.append(ip)
            ip_list.append(tmp_rt)
            
        return pd.DataFrame( {'ip' : ip_target, 'dstn_ip' : ip_list })
        

    # to_from -> 2019-12-22 00:00:00 
    # to_date -> 2019-12-21 00:00:00         
    def task_ip_trend(self, from_date, to_date):
        print ("Start task ip trend")
        #########################################################################################
        ################################ Data Load ##############################################
        # from_date, to_date 데이터를 가져온다. 
        # 쿼리 :  log_code:(110401 110402) AND eqp_ip:(장비IP) 
        #         AND dstn_port:53 AND ingres_if:(eth11 down2223) AND prtc:UDP
        df = pd.read_csv(os.path.join(cst.PATH_DATA, 'mw_nsims_type1_fw_.csv'))
        df.rename(columns = self.nsims_dict, inplace = True)
        # 데이터에서 조건에 맞는 날짜 데이터를 가져온다. 
        # Malware는 하루 배치
        # to_from -> 2019-12-22 00:00:00 
        # to_date -> 2019-12-21 00:00:00 
        df['gather_time'] = df['gather_time'].astype(str)
        df['date'] = df['gather_time'].apply(lambda x: parse(x).strftime("%Y%m%d"))
        df = df.query(" date < '{}' and  date >= '{}' ".format(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d")))
        #########################################################################################
        #########################################################################################

        agg_rt = df.groupby(['src_ip', 'dstn_ip']).agg({'action' : 'count'}).reset_index()
        
        #--> 전처리2 (대전FW 1일 단위 내부 IP 모니터링 결과)
        # 날짜, 내부IP, DNS IP 
        preprocess_2_1 = agg_rt[['src_ip', 'dstn_ip']]
        preprocess_2_1['date'] = to_date.strftime("%Y%m%d")
        preprocess_2_1 = preprocess_2_1[['date', 'src_ip', 'dstn_ip']].rename(columns = {"dstn_ip" : "dns_ip", 'src_ip' : 'int_ip'})


        #--> 전처리2 가공하여 1일 단위 집계
        dns_list_df = pd.DataFrame(agg_rt.groupby(['src_ip'])['dstn_ip'].apply(list)).reset_index()
        dns_list_df_count = agg_rt.groupby(['src_ip'])[['dstn_ip']].count().reset_index()
        preprocess_2_2 = pd.merge(dns_list_df_count, dns_list_df, on='src_ip')
        preprocess_2_2['date'] = to_date.strftime("%Y%m%d")
        preprocess_2_2 = preprocess_2_2[['date', 'src_ip', 'dstn_ip_x', 'dstn_ip_y']].rename(columns = {"dstn_ip_x" : "dns_count", 'src_ip' : 'int_ip', 'dstn_ip_y' : 'dns_ip_list'})

        

        #--> 전처리2 가공하여 7일 단위 집계
        ######
        # date	날짜	x	집계 날짜(00:00 ~ 24:00)
        # int_ip_count	DNS 개수	x	내부IP에서 해당 날짜에 접속한 유니크한 DNS 수
        ###### 7일치 데이터를 가져옴, 구조는 preprocess_2_2 테이블 형식의 데이터를 가져와서 preprocess_2_2 변수에 입력하고 
        ###### pivot해서 집계함
        preprocess_2_3 = df.groupby(['dstn_ip', 'src_ip'])[['action']].count()
        preprocess_2_3.reset_index(inplace=True)
        preprocess_2_3 = preprocess_2_3.groupby(['dstn_ip'])[['src_ip']].count().reset_index()
        preprocess_2_3['date'] = to_date.strftime("%Y%m%d")
        preprocess_2_3.rename(columns = {'dstn_ip' : 'ex_dns_ip', 'src_ip' : 'int_ip_count'}, inplace=True)
        preprocess_2_3['status'] = ""
        preprocess_2_3 = preprocess_2_3[['date', 'ex_dns_ip', 'int_ip_count', 'status']]

        
        return [preprocess_2_1, preprocess_2_2, preprocess_2_3]
        

    # 외부로 callback하는 외부 DNS 이벤트 쿼리
    def task_model_type_1(self, from_date, to_date):
        
        #########################################################################################
        ################################ Data Load ##############################################
        # from_date, to_date 데이터를 가져온다. 
        # 쿼리 : log_code:190102 AND prtc:(HTTP UDP FTP TCP) AND direction:inbound AND detect_category:/.+/ AND object_type:callback
        df = pd.read_csv(os.path.join(cst.PATH_DATA, 'mw_nsims_type_1_01.csv'))
        df.rename(columns = self.nsims_dict, inplace = True)
        # 데이터에서 조건에 맞는 날짜 데이터를 가져온다. 
        # Malware는 하루 배치
        # to_from -> 2019-12-22 00:00:00 
        # to_date -> 2019-12-21 00:00:00 
        df['gather_time'] = df['gather_time'].astype(str)
        df['date'] = df['gather_time'].apply(lambda x: parse(x).strftime("%Y%m%d"))
        df = df.query(" date < '{}' and  date >= '{}' ".format(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d")))
        #########################################################################################
        #########################################################################################

        # 사용할 데이터만 추출 
        ###-> type 1에 맞는 데이터 필터링 
        # 190102 로그데이터 사용 
        df = df[df['log_code'] == 190102]
        
        # type 1에서는 아래 4개의 프로토콜의 데이터만 사용한다. 
        target_prtc = ['HTTP', 'UDP', 'FTP', 'TCP']
        df = df[df['prtc'].apply(lambda x : x in target_prtc)]
        
        # direction의 값이 inbound인 값 
        
        df = df[df['direction'] == 'inbound']

        # detect_category 값이 Null 값이 아닌 데이터 
        df = df[df['detect_category'].notnull()]

        # object_type의 값이 callback인 데이터 
        df = df[df['object_type'] == 'callback']

        # 확인 필요한 외부 DNS IP 추출 
        df_result = df['attacker_ip'].tolist()

        #####
        # 외부 DNS IP 추출하여 모니터링 DB와 비교
        # -7일치 트렌드 분석 후 나온 외부 DNS주소     
        # trend_from   --> 2019-12-22
        # trend_to     --> 2019-12-15
        trend_from, trend_to = mw_set_date(from_date.strftime("%Y-%m-%d"), 7)

        # trend 분석 후 대전, 국통망의 공통된 목적지 IP 추출 
        rt = self.type_1_trend_analysis(trend_from, trend_to)
    

        final_rt = list(set(df_result) & set(rt))
        
        # 만약 교집합 IP가 있으면 해당 로그키값, 수집시간, IP만 따로 df 생성 
        mask = df['attacker_ip'].apply(lambda x: x in final_rt)
        
        rt_df = df.loc[mask, ['key', 'attacker_ip', 'attack_target']]
        rt_df.rename(columns = { 'attacker_ip' : 'ip_res',
                                'attack_target' : 'res_ip'}, inplace=True)

        return rt_df


    def task_model_type_2(self, from_date, to_date):
        
        #########################################################################################
        ################################ Data Load ##############################################
        # from_date, to_date 데이터를 가져온다. 
        # 쿼리 : log_code:190102 AND prtc:(HTTP UDP FTP TCP) AND direction:inbound AND detect_category:/.+/ AND object_type:file
        df = pd.read_csv(os.path.join(cst.PATH_DATA, 'mw_nsims_type_2_01.csv'))
        df.rename(columns = self.nsims_dict, inplace = True)

        # 데이터에서 조건에 맞는 날짜 데이터를 가져온다. 
        # Malware는 하루 배치
        # from_date -> 2019-12-22 00:00:00 
        # to_date   -> 2019-12-21 00:00:00 
        df['gather_time'] = df['gather_time'].astype(str)
        df['date'] = df['gather_time'].apply(lambda x: parse(x).strftime("%Y%m%d"))
        df = df.query(" date < '{}' and  date >= '{}' ".format(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d")))
        #########################################################################################
        #########################################################################################



        ###-> type 1에 맞는 데이터 필터링 
        # 190102 로그데이터 사용 
        df_type_2 = df[df['log_code'] == 190102]
        
        # type 1에서는 아래 4개의 프로토콜의 데이터만 사용한다. 
        target_prtc = ['HTTP', 'UDP', 'FTP', 'TCP']
        df_type_2 = df_type_2[df_type_2['prtc'].apply(lambda x : x in target_prtc)]
        
        # direction의 값이 inbound인 값 
        df_type_2 = df_type_2[df_type_2['direction'] == 'inbound']

        # detect_category 값이 Null 값이 아닌 데이터 
        df_type_2 = df_type_2[df_type_2['detect_category'].notnull()]

        # object_type의 값이 file인 데이터 
        df_type_2 = df_type_2[df_type_2['object_type'] == 'file']

        # 내부 IP 추출 
        df_result = df_type_2['attack_target'].unique().tolist()

        ############
        ## --> df_result의 IP들의 방화벽 형태 분석 
        ## IP별 전날 접속한 외부 IP와 당일 외부 IP에서의 신규IP 
        
        
        # 현재 프로그램이 실행된 날짜 '
        # 
        #today = date.today()
        # pseudo code에서는  type2는 실행한 날짜를 2019-12-22 00:00:00일이라고 가정한다. 
        # 정시에 배치가 돌 때, 2019-12-21, 2019-12-20일의 외부 접속 IP 접속 목록에서 
        # 신규 IP만 추출한다. 
        
        # to_date   -> 2019-12-21 00:00:00 
        target_from, target_to = mw_set_date(to_date.strftime("%Y-%m-%d"))
        # target_from -> 2019-12-21 00:00:00 
        # target_to   -> 2019-12-20 00:00:00 
        #
        # 검색 IP를 하나씩 가져와서 type_2_query_list에 전달 
        rt_df = pd.DataFrame()
        for x in df_result:
            # 현재 날짜에서 두번째 날짜까지 신규 IP 추출 
            # 기본값으로 전날으로 한다. 
            rt = self.type_2_query_diff(x, target_from, target_to)
            rt_df = rt_df.append(pd.DataFrame( {'attack_target' : x, 'res_ip' : [rt]}))
            
        mask = df_type_2['attack_target'].apply(lambda x: x in rt_df['attack_target'].tolist())
        rt = df_type_2.loc[mask, ['key', 'attack_target']]
        
        rt2 = pd.merge(rt, rt_df, on='attack_target')
        
        rt2.rename(columns = { 'attack_target' : 'ip_res'}, inplace=True)
        return rt2
            

    def task_model_type_3(self, from_date, to_date):

        #########################################################################################
        ################################ Data Load ##############################################
        # from_date, to_date 데이터를 가져온다. 
        # 쿼리 : log_code:190102 AND prtc:(HTTP UDP FTP TCP) AND direction:inbound AND detect_category:/.+/ AND object_type:file
        df = pd.read_csv(os.path.join(cst.PATH_DATA, 'mw_nsims_type_3_01.csv'))
        df.rename(columns = self.nsims_dict, inplace = True)

        # 데이터에서 조건에 맞는 날짜 데이터를 가져온다. 
        # Malware는 하루 배치
        # from_date -> 2019-12-22 00:00:00 
        # to_date   -> 2019-12-21 00:00:00 
        df['gather_time'] = df['gather_time'].astype(str)
        df['date'] = df['gather_time'].apply(lambda x: parse(x).strftime("%Y%m%d"))
        df = df.query(" date < '{}' and  date >= '{}' ".format(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d")))
        #########################################################################################
        #########################################################################################


        # 190102 로그데이터 사용 
        df_type_3 = df[df['log_code'] == 190102]
        
        # type 1에서는 아래 4개의 프로토콜의 데이터만 사용한다. 
        target_prtc = ['HTTP', 'UDP', 'FTP', 'TCP']
        df_type_3 = df_type_3[df_type_3['prtc'].apply(lambda x : x in target_prtc)]
        
        # direction의 값이 outside인 값 
        df_type_3 = df_type_3[df_type_3['direction'] == 'outside']

        # detect_category 값이 Null 값이 아닌 데이터 
        df_type_3 = df_type_3[df_type_3['detect_category'].notnull()]

        # mal_vt_sum의 내용에 YARA가 포함된 값만 추출 
        df_type_3 = df_type_3[df_type_3['mal_vt_sum'].apply(lambda x : True if str(x).find("YARA") >= 0 else False)]

        # session_info에 공격 대상 IP가 들어있어 추출함 
        pattern = "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        df_type_3['src_ip'] = df_type_3['session_info'].apply(lambda x : re.findall(pattern, str(x).split("->")[-1])[0])

        # 대상 IP 추출 
        target = df_type_3[['gather_time', 'src_ip','key']].copy()
        
        # 대상 IP에서 
        ######## 
        ### --> 대상IP의 공통으로 나오는 목적지 IP 추출 
        
        # 해당 IP의 Outbound 통신 결과 
        # from_date -> 2019-12-22 00:00:00 
        # to_date   -> 2019-12-21 00:00:00
        result = self.type_3_query_dstn_ip(target, from_date, to_date)

        
        # 외부로 접속한 IP가 공통적인게 있는지 확인 
        # 교집합 
        first = set(result['dstn_ip'].iloc[0])
        for ip in result['dstn_ip'].iloc[1:]:
            set_rt = first & set(ip)
            
        
        target['res_ip'] = str(list(set_rt))
        target = target[['key', 'res_ip', 'src_ip']]
        target.rename(columns={'src_ip' : "ip_res"}, inplace=True)
        return target

    def get_data_df(self, load_file_dir, attack_type):

        file = load_file_dir.format(attack_type)
        
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_csv(file, encoding="utf-8")
        df.rename(columns = self.nsims_dict, inplace = True)

        return df

    def get_save_df(self, save_file_dir, type_1, type_2, type_3):
        final_df = pd.concat([type_1, type_2, type_3])
        final_df.reset_index(drop=True,inplace=True)    
        final_df.to_csv(save_file_dir)
        return final_df

    def get_save_trend(self, save_file_dir, df_list):
        for idx, df in enumerate(df_list):
            df.to_csv(os.path.join(save_file_dir, "mw_iptrend_{}_01.csv".format(idx)))

