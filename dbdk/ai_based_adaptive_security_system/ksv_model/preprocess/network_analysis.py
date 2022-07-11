# -*- coding: utf-8 -*-
# network analysis algorithms

import pandas as pd
import numpy as np
import networkx as nx
import operator
from ksv_model.preprocess.common import filter_white_ip


class NetworkAnalysis():

    def __init__(self):
        self.filter_wip = filter_white_ip()

    def _draw_network_graph(self, df_ip):
        DG = nx.DiGraph()
        df_nx = df_ip[:].copy()  # 그래프 그릴 범위 추출
        src_ip_lst = df_nx['src_ip'].unique().tolist()
        dstn_ip_lst = df_nx['dstn_ip'].unique().tolist()
        ip_nodes = list(set(src_ip_lst + dstn_ip_lst))
        DG.add_nodes_from(ip_nodes)
        for _, row in df_nx.iterrows():
            DG.add_edge(row['src_ip'], row['dstn_ip'], weight=row['count'])
        
        return DG

    def _weighted_degree_centrality(self, g, normalized=False):
        w_d_cent = {n:0.0 for n in g.nodes()}
        for u, v, d in g.edges(data=True):
            w_d_cent[u] += d['weight']
            w_d_cent[v] += d['weight']
        if normalized==True:
            weighted_sum = sum(w_d_cent.values())
            return {k:v/weighted_sum for k, v in w_d_cent.items()}
        else:
            return w_d_cent

    def _filter_int_ip(self, dict_items):
        dict_items_cp = dict_items.copy()
        df_tmp_ip = pd.DataFrame(list(dict_items_cp.keys()), columns=['ip'])
        # exclude int ip
        mask = self.filter_wip(df_tmp_ip['ip'])
        df_int_ip = df_tmp_ip[mask]
        int_ip_list = df_int_ip['ip'].unique().tolist()
        for ip in dict_items_cp.keys():
            if ip in int_ip_list:
                del dict_items[ip]
        return dict_items

    def get_high_degree_centrality_ip(self, df_ip, cd_threshold=0.1):
        dg = self._draw_network_graph(df_ip)
        
        cd = self._filter_int_ip(nx.out_degree_centrality(dg))
        sorted_cd = sorted(cd.items(), key=operator.itemgetter(1), reverse=True)
        
        df_sorted_cd = pd.DataFrame(sorted_cd, columns=['src_ip', 'cd'])
        
        # source IPs whose degree centrality value is over cd_threshold(0.1)
        df_cd = df_sorted_cd[df_sorted_cd['cd'] > cd_threshold]
        if df_cd is not None and not df_cd.empty:
            if df_cd.shape[0] > 3:
                res_cd = df_cd[0:3]['src_ip'].unique().tolist()
            else:
                res_cd = df_cd['src_ip'].unique().tolist()
        
        wcent = self._filter_int_ip(self._weighted_degree_centrality(dg, True))
        sorted_wcent = sorted(wcent.items(), key=operator.itemgetter(1), reverse=True)
        
        df_sorted_wcd = pd.DataFrame(sorted_wcent, columns=['src_ip', 'wcd'])
                
        df_wcd = df_sorted_wcd[df_sorted_wcd['wcd'] > cd_threshold]
        
        if df_wcd is not None and not df_wcd.empty:
            if df_wcd.shape[0] > 3:
                res_wcd = df_wcd[0:3]['src_ip'].unique().tolist()
            else:
                res_wcd = df_wcd['src_ip'].unique().tolist()

        return list(set(res_cd + res_wcd))
