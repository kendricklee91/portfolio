# -*- coding: utf-8 -*-
# clustering algorithms

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class KMeansClustering():

    def __init__(self):
        pass

    def _find_k_value(self, max_k, df_input):
        # find k
        ks = range(2, max_k)
        inertias = []

        for k in ks:
            model_tmp = KMeans(n_clusters=k, algorithm='auto')
            model_tmp.fit(df_input)
            inertias.append(model_tmp.inertia_)

        powers = np.log10(np.array(inertias)).astype(int)
        power = np.argmax(np.bincount(powers))

        inertias_diff = []
        for idx, i in enumerate(inertias):
            if idx == 0:
                continue
            inertias_diff.append((inertias[idx-1] - i)/(10**power))

        for idx, idiff in enumerate(inertias_diff):
            if idiff < 1:
                k = idx + 2
                break

        return k

    def _fit_clustering(self, k, df_input):
        scaler = StandardScaler()
        model2 = KMeans(n_clusters=k, algorithm='auto')
        pipeline = make_pipeline(scaler, model2)
        pipeline.fit(df_input)

        df_clustered = pd.DataFrame(pipeline.predict(df_input))
        df_clustered.columns = ['cluster']

        ##### get metrics
        # Silhouette Coefficient
        # 값의 범위는 (-1, 1)이며 1에 가까울수록 잘 된 클러스터링. 0 근처는 오버래핑 클러스터.
        sc_score = metrics.silhouette_score(df_input, df_clustered['cluster'], metric='euclidean')
        # Calinski-Harabasz Index
        # 클러스터 밀집도가 높고 잘 분리되어있을 때 값이 높음
        ch_score = metrics.calinski_harabaz_score(df_input, df_clustered['cluster'])
        # Davies-Bouldin Index
        # 값이 0에 가까울수록 잘 된 클러스터링
        db_score = metrics.davies_bouldin_score(df_input, df_clustered['cluster'])

        score = { 
            "silhouette_coefficient" : float(sc_score),
            "calinski_harabasz_index" : float(ch_score),
            "davies_bouldin_index" : float(db_score)
        }

        return df_clustered, dict(score)