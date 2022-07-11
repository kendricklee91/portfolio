# -*- coding: utf-8 -*-
# dataset recommendation model for re-training payload based models

import os
import pandas as pd

import tensorflow  as tf
import keras.backend.tensorflow_backend as K
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.model_selection import train_test_split

from ksv_model.preprocess.conv_payload import KnownAttackPreprocess
from ksv_model.preprocess.clustering import KMeansClustering
import ksv_model.config.gpu_dev as gd
import ksv_model.preprocess.common as cm


DIM_INPUT = 1024
DIM_ENCODED = DIM_INPUT//4
MAX_PAD_SIZE = 1024


# Post-processing for re-train: logs without payload - Information Gathering
class ModelDSRbase:
    def __init__(self):
        pass

    def recommend_retrain_dataset(self, attack_type, df, threshold):
        if not self._validate_param(attack_type, df):
            return False

        df = self._process_result(df)

        # select dataset cluster to re-train
        return self._select_dataset(df, threshold), None

    def _validate_param(self, attack_type, df):
        # check attack_type, df.columns, dim
        return True

    def _process_result(self, df):
        df['attack_nm'] = df['attack_nm'].astype(str)
        #df.loc[:, 'action_val'] = df['action'].apply(cm.convert_action_value)

        #df['0'] = df['predict'].apply(lambda x: x[0][0])
        #df['1'] = df['predict'].apply(lambda x: x[0][1])
        #df['label_inf'] = df.apply(lambda x: 0 if x['0'] > x['1'] else 1, axis=1)
        #df['prob_inf'] = df.apply(lambda x: x['0'] if x['0'] > x['1'] else x['1'], axis=1)

        return df

    def _select_dataset(self, df, threshold):
        ### priority value for dataset recommendation
        # 1: classification probability based
        # 50: attack_nm based
        # 999: n/a

        # recommend attack_nm list to reexamine the IPS signature
        df_d = df[df['action_val']==1]  # denied by IPS
        df_d_attack = df_d[df_d['label_inf']==0][['attack_nm', 'prob_inf']].groupby(['attack_nm']
                            ).mean().reset_index().sort_values('prob_inf', ascending=True).reset_index()
        attack_nm_rvrt = df_d_attack[df_d_attack['prob_inf'] < threshold[0]]['attack_nm'].unique().tolist()  # signature review & re-train
        attack_nm_rv = df_d_attack[df_d_attack['prob_inf'] >= threshold[0]]['attack_nm'].unique().tolist() # signature review only
        attack_nm_check_policy = list(set(attack_nm_rvrt)|set(attack_nm_rv))

        if attack_nm_check_policy is not None and len(attack_nm_check_policy) > 0:
            df['check_policy'] = df['attack_nm'].apply(lambda x: 1 if x in attack_nm_check_policy else 0)
        else:
            df['check_policy'] = 0

        priority = 50
        if attack_nm_rvrt is not None and len(attack_nm_rvrt) > 0:
            df['rcmnd_priority'] = df.apply(lambda x: priority if x['attack_nm'] in attack_nm_rvrt else x['rcmnd_priority'], axis=1)

        # recommend logs to re-labeling and re-training
        priority = 1
        df_a = df[df['action_val']==0]  # alarm by IPS or IDS
        df_retrain = df_a[df_a['prob_inf'] < threshold[0]]
        df.loc[(df['action_val']==0)&(df['prob_inf'] < threshold[0]), 'rcmnd_priority'] = priority

        return df, attack_nm_rvrt, attack_nm_rv, df_retrain, None


# Post-processing for re-train: logs with payload - SQLi/XSS/RCE/UAA/FUP/FDO
class ModelDSR(ModelDSRbase):
    def __init__(self, gpu_dev=None):
        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev) 


    def recommend_retrain_dataset(self, attack_type, df, threshold):
        if not self._validate_param(attack_type, df):
            return False

        df = self._process_result(df)
        df = self._process_payload(attack_type, df)

        self._gpu_context_enter()
        autoencoder, encoder, decoder = self._init_autoencoder()
        df_enp = self._fit_autoencoder(df, autoencoder, encoder, decoder)
        self._gpu_context_exit()

        df, score = self._clustering_payload(df, df_enp)

        # select dataset cluster to re-train
        return self._select_dataset(df, threshold), score


    def _process_payload(self, attack_type, df):
        kap = KnownAttackPreprocess()

        # suppose that we already have inference completed dataset, df
        df = df[~df['payload'].isnull()]  # leave only logs with payload
        df['payload'] = df['payload'].astype(str)

        #df['payload_ps'] = df['payload'].apply(lambda x: kap.preprocess_payload_str(attack_type, x))
        df['payload_ps_pad'] = df.apply(lambda x: kap.asc_padding(x['payload_ps'], MAX_PAD_SIZE, x['label'])[1:DIM_INPUT+1], axis=1)

        return df

    def _gpu_context_enter(self):
        self.context = K.tf.device(self.gpu_dev)
        self.context.__enter__()

    def _gpu_context_exit(self):
        self.context.__exit__(None, None, None)

    def _init_autoencoder(self):
        input_dim = DIM_INPUT
        encoding_dim = DIM_ENCODED
        unit = (input_dim - encoding_dim)//3

        input_stream = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim + unit*2, activation='relu')(input_stream)
        encoded = Dense(encoding_dim + unit, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        decoded = Dense(encoding_dim + unit, activation='relu')(encoded)
        decoded = Dense(encoding_dim + unit*2, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_stream, decoded)

        # create encoder
        encoder = Model(input_stream, encoded)

        # create decoder
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-3](encoded_input)
        decoder_layer = autoencoder.layers[-2](decoder_layer)
        decoder_layer = autoencoder.layers[-1](decoder_layer)
        decoder = Model(encoded_input, decoder_layer)

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return autoencoder, encoder, decoder

    def _fit_autoencoder(self, df, autoencoder, encoder, decoder):
        x = df[['payload_ps_pad']]
        x_ser = pd.Series(x['payload_ps_pad'].tolist(), index=x.index)
        x_ps = pd.DataFrame(x_ser.values.tolist(), index=x_ser.index)

        x_train, x_test = train_test_split(x_ps, test_size = 0.1, random_state = 1)
        x_train = x_train.astype('float32')/255.
        x_test = x_test.astype('float32')/255.

        autoencoder.fit(x_train, x_train, epochs=15, batch_size=500, shuffle=True, validation_data=(x_test, x_test))

        #encoded_stream = encoder.predict(x_test)
        #decoded_stream = decoder.predict(encoded_stream)
        encoded_payload = encoder.predict(x_ps.astype('float32')/255.)

        df_enp = pd.DataFrame(encoded_payload, index=x_ser.index)

        return df_enp

    def _clustering_payload(self, df, df_enp):
        num_attack = len(df['attack_nm'].unique().tolist())
        max_k = 50
        if num_attack < 20:
            max_k = 20
        else:
            max_k = num_attack // 2
        
        kmeans = KMeansClustering()
        k = kmeans._find_k_value(max_k, df_enp)
        
        df_cluster, score = kmeans._fit_clustering(k, df_enp)
        print('clustering score: ', score)

        df = pd.concat([df, df_cluster], axis=1)

        return df, score

    def _select_dataset(self, df, threshold):
        ### priority value for dataset recommendation
        # 1: classification probability based
        # 50: attack_nm based
        # 100~: cluster based
        # 999: n/a

        # recommend certain cluster of logs based on payload similarity
        df['rcmnd_priority'] = 999
        df_ds = df[['cluster', 'prob_inf']].groupby(['cluster']).mean().reset_index()
        df_ds_sort = df_ds.sort_values('prob_inf', ascending=True).reset_index()
        k_rcmnd = df_ds_sort[df_ds_sort['prob_inf'] < threshold[0]]['cluster'].unique().tolist()
        priority = 100

        if not k_rcmnd:
            print('No log cluster to recommend.')
            df_cluster = None
        else:
            for k in k_rcmnd:
                df.loc[df['cluster']==k, 'rcmnd_priority'] = priority
                priority += 1

            df_cluster = df[df['rcmnd_priority']!=999]

        # recommend attack_nm list to reexamine the IPS signature
        df_d = df[df['action_val']==1]  # denied by IPS
        df_d_attack = df_d[df_d['label_inf']==0][['attack_nm', 'prob_inf']].groupby(['attack_nm']
                            ).mean().reset_index().sort_values('prob_inf', ascending=True).reset_index()
        attack_nm_rvrt = df_d_attack[df_d_attack['prob_inf'] < threshold[0]]['attack_nm'].unique().tolist()  # signature review & re-train
        attack_nm_rv = df_d_attack[df_d_attack['prob_inf'] >= threshold[0]]['attack_nm'].unique().tolist() # signature review only
        attack_nm_check_policy = list(set(attack_nm_rvrt)|set(attack_nm_rv))

        if attack_nm_check_policy is not None and len(attack_nm_check_policy) > 0:
            df['check_policy'] = df['attack_nm'].apply(lambda x: 1 if x in attack_nm_check_policy else 0)
        else:
            df['check_policy'] = 0

        priority = 50
        if attack_nm_rvrt is not None and len(attack_nm_rvrt) > 0:
            df['rcmnd_priority'] = df.apply(lambda x: priority if x['attack_nm'] in attack_nm_rvrt else x['rcmnd_priority'], axis=1)
        
        # recommend logs to re-labeling and re-training
        priority = 1
        df_a = df[df['action_val']==0]  # alarm by IPS or IDS
        df_retrain = df_a[df_a['prob_inf'] < threshold[0]]
        df.loc[(df['action_val']==0)&(df['prob_inf'] < threshold[0]), 'rcmnd_priority'] = priority

        return df, attack_nm_rvrt, attack_nm_rv, df_retrain, df_cluster
