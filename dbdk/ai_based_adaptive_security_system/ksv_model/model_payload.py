from __future__ import print_function

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Reshape, Flatten

from keras.models import Sequential, model_from_json, load_model
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils, multi_gpu_model
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping
from keras import optimizers

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# from hyperopt import Trials, STATUS_OK, tpe # for test

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from ksv_model.preprocess.conv_payload import KnownAttackPreprocess
import ksv_model.config.gpu_dev as gd
import ksv_model.config.const as cst
import ksv_model.preprocess.common as cm

import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import datetime
import pprint
import keras
import time
import json
import csv
import os
import math

import pickle

DIM_INPUT = 1028

def gpu_context_enter(context, gpu_dev):
    context = K.tf.device(gpu_dev)
    context.__enter__()

    return context

def gpu_context_exit(context):
    context.__exit__(None, None, None)


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

def task_post_process_auto_block(df, model_type, th_auto):
    # 1차 고도화 모델 6개 및 정보수집 모델의 추론 데이터 기반으로 자동 차단 대상 식별

    # 1. 일차 식별 기준은 threshold.json의 모델별 upper value를 활용
    th_val = load_threshold_value(model_type=model_type, th_auto=th_auto)
    df_auto_block = df[(df['model_type']==model_type)&(df['action_val']==0)&(df['label_inf']==1)&(df['prob_inf']>=th_val[1])]

    # 2. 일차 식별된 로그 중 출발지IP가 내부 IP인 경우는 필터링하여 자동 차단 대상에서 제외
    #call interal_ip_filtering_function() here

    # 3. 자동 차단 대상은 AI 플랫폼에서 적응형 보안관리 시스템으로 REST API 방식으로 전달
    #call API here

    return df_auto_block

"""
Normal train model
--------------------------
1. Data load & split train and test
3. Create model and train
4. Return model, history, loss, accuracy
"""
class known_model:
    def __init__(self, gpu_dev=None):
        self.dls = data_load_save()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev) 

    def create_model(self, load_file_dir, attack_type, params):      
        gpu_context = gpu_context_enter(self.context, self.gpu_dev)

        # load data
        dataset, x_train, y_train, x_test, y_test = self.dls.get_data_file(load_file_dir, attack_type)
        # print(dataset.shape) # for test

        model = Sequential()
        model.add(Reshape((dataset.shape[1] - 1, 1), input_shape = (dataset.shape[1] - 1, )))

        # conv-activate-pooling #1
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size'], padding = params['padding']))
        model.add(Activation(params['activation']))
        model.add(MaxPooling1D(params['pooling_size']))

        # conv-activate-pooling #2
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size'], padding = params['padding']))
        model.add(Activation(params['activation']))
        model.add(MaxPooling1D(params['pooling_size']))

        # conv-activate-pooling #3
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size'], padding = params['padding']))
        model.add(Activation(params['activation']))
        model.add(MaxPooling1D(params['pooling_size']))

        model.add(Dropout(params['dropout']))
        model.add(Flatten())
        
        model.add(Dense(params['dense_layer_1']))
        model.add(Activation(params['activation']))

        model.add(Dense(params['dense_layer_2']))
        model.add(Activation(params['activation']))

        # fix layer
        model.add(Dense(params['dense_fix']))
        model.add(Activation(params['activation_fix']))

        model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])
        hist = model.fit(x_train, y_train, batch_size = params['batch_size'], epochs = params['epochs'],
                            verbose = params['verbose'], validation_data = (x_test, y_test), shuffle = params['shuffle'])
        gpu_context_exit(gpu_context)
        
        eval_loss, eval_acc = model.evaluate(x_test, y_test)
        # print('loss :', eval_loss, ', acc :', eval_acc) # for test

        return model, hist, eval_loss, eval_acc

"""
Re-train model
-------------------
1. Data load & split train and test
2. Load pre-trained model and weight
3. Trails of best hyper-parameter search (use parameter : optimizer, batch size)
5. Return best hyper-parameter, best model, accuracy
"""
class known_model_retrain:
    def __init__(self, gpu_dev=None):
        self.dls = data_load_save()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev)

    def retrain_model(self, params):
        # load data
        dataset, x_train, y_train, x_test, y_test = self.dls.get_data_file(params['load_file_dir'], params['attack_type'])

        gpu_context = gpu_context_enter(self.context, self.gpu_dev)

        # load model
        load_model = self.dls.get_load_model_file(params['load_model_dir'], params['load_model_wt_dir'], params['attack_type'])
        
        # Training 
        load_model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])
        hist = load_model.fit(x_train, y_train, batch_size = params['batch_size'], epochs = params['epochs'],
                            verbose = params['verbose'], validation_data = (x_test, y_test), shuffle = params['shuffle'])

        gpu_context_exit(gpu_context)

        validation_acc = np.max(hist.history['val_acc'])

        return {'loss' : -validation_acc, 'status' : STATUS_OK, 'model' : load_model, 'x_test' : x_test, 'y_test' : y_test}

    def optimize(self, load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type):
        space       = self._get_space_retrain(load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type)

        trials      = Trials()
        best_run    = fmin(self.retrain_model, space, algo = tpe.suggest, max_evals = space['max_evals'], trials = trials)

        best_params = space_eval(space, best_run)
        best_model  = trials.best_trial['result']['model']
        best_trial  = trials.best_trial

        x_test = best_trial['result']['x_test']
        y_test = best_trial['result']['y_test']

        return self._get_best_model(best_params, best_model, best_trial, x_test, y_test)

    def _get_space_retrain(self, load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type):
        # Set parameter search bounds
        space = {
            'optimizer'      : hp.choice('optimizer', params['param_range']['optimizer_range']),
            'batch_size'     : hp.choice('batch_size', params['param_range']['batch_range'])
        }

        space['attack_type']            = attack_type
        space['load_file_dir']          = load_file_dir
        space['load_model_dir']         = load_model_dir
        space['load_model_wt_dir']      = load_model_wt_dir

        space['loss']                   = params['loss']
        space['epochs']                 = params['epochs']
        space['verbose']                = params['verbose']
        space['shuffle']                = params['shuffle']
        space['metrics']                = params['metrics']
        space['max_evals']              = int(params['max_evals'])

        return space

    def _get_best_model(self, b_params, b_model, b_trial, x_test, y_test):
        best_params = b_params
        best_model  = b_model
        best_trial  = b_trial

        model_json  = best_model.to_json()
        data        = json.loads(model_json)

        loss, acc = best_model.evaluate(x_test, y_test)

        return best_params, best_model, acc


"""
Hyper-parameter auto-optimization model
-----------------------------------
1. Data load & split train and test
2. Trails of best hyper-parameter search
3. Return best hyper-parameter, best model, accuracy
"""
class known_model_hopt:
    def __init__(self, gpu_dev=None):
        self.dls = data_load_save()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev)

    def create_model(self, params):
        # load data
        dataset, x_train, y_train, x_test, y_test = self.dls.get_data_file(params['load_file_dir'], params['attack_type'])

        gpu_context = gpu_context_enter(self.context, self.gpu_dev)

        model = Sequential()
        model.add(Reshape((dataset.shape[1] - 1, 1), input_shape = (dataset.shape[1] - 1, )))

        # conv-activate-pooling #1
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size_1'], padding = params['padding'], name = 'kernelsize_1'))
        model.add(Activation(params['activation_1'], name = 'activation_1'))
        model.add(MaxPooling1D(params['pooling_size_1'], name = 'maxpooling_1'))

        # conv-activate-pooling #2
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size_2'], padding = params['padding'], name = 'kernelsize_2'))
        model.add(Activation(params['activation_2'], name = 'activation_2'))
        model.add(MaxPooling1D(params['pooling_size_2'], name = 'maxpooling_2'))

        # conv-activate-pooling #3
        model.add(Convolution1D(filters = params['filter_size'], kernel_size = params['kernel_size_3'], padding = params['padding'], name = 'kernelsize_3'))
        model.add(Activation(params['activation_3'], name = 'activation_3'))
        model.add(MaxPooling1D(params['pooling_size_3'], name = 'maxpooling_3'))

        model.add(Dropout(params['dropout'], name = 'dropout'))
        model.add(Flatten(name = 'flatten'))

        model.add(Dense(params['dense_size_1'], name = 'dense_1'))
        model.add(Activation(params['activation_4'], name = 'activation_4'))

        model.add(Dense(params['dense_size_2'], name = 'dense_2'))
        model.add(Activation(params['activation_5'], name = 'activation_5'))

        model.add(Dense(params['dense_fix'], name = 'dense_3'))
        model.add(Activation(params['activation_fix'], name = 'activation_6'))

        model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])
        hist = model.fit(x_train, y_train, batch_size = params['batch_size'], epochs = params['epochs'],
                            verbose = params['verbose'], validation_data = (x_test, y_test), shuffle = params['shuffle'])
                            
        gpu_context_exit(gpu_context)
        
        validation_acc = np.max(hist.history['val_acc'])
        print('Best validation acc of epoch : ', validation_acc) # for test

        return {'loss' : -validation_acc, 'status' : STATUS_OK, 'model' : model, 'x_test' : x_test, 'y_test' : y_test}
    
    def optimize(self, load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type):
        space       = self._get_space_hopt(load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type)

        trials      = Trials()
        best_run    = fmin(self.create_model, space, algo = tpe.suggest, max_evals = space['max_evals'], trials = trials)

        best_params = space_eval(space, best_run)
        best_model  = trials.best_trial['result']['model']
        best_trial  = trials.best_trial

        x_test = best_trial['result']['x_test']
        y_test = best_trial['result']['y_test']

        return self._get_best_model(best_params, best_model, best_trial, x_test, y_test)

    def _get_space_hopt(self, load_file_dir, load_model_dir, load_model_wt_dir, params, attack_type):
        # Set parameter search bounds
        space = {
                    'kernel_size_1'  : hp.choice('kernel_size_1', params['param_range']['kernel_range']),
                    'kernel_size_2'  : hp.choice('kernel_size_2', params['param_range']['kernel_range']),
                    'kernel_size_3'  : hp.choice('kernel_size_3', params['param_range']['kernel_range']),
                    'pooling_size_1' : hp.choice('pooling_size_1', params['param_range']['pooling_range']),
                    'pooling_size_2' : hp.choice('pooling_size_2', params['param_range']['pooling_range']),
                    'pooling_size_3' : hp.choice('pooling_size_3', params['param_range']['pooling_range']),
                    'dense_size_1'   : hp.choice('dense_size_1', params['param_range']['dense_range']),
                    'dense_size_2'   : hp.choice('dense_size_2', params['param_range']['dense_range']),
                    'activation_1'   : hp.choice('activation_1', params['param_range']['activation_range']),
                    'activation_2'   : hp.choice('activation_2', params['param_range']['activation_range']),
                    'activation_3'   : hp.choice('activation_3', params['param_range']['activation_range']),
                    'activation_4'   : hp.choice('activation_4', params['param_range']['activation_range']),
                    'activation_5'   : hp.choice('activation_5', params['param_range']['activation_range']),
                    'optimizer'      : hp.choice('optimizer', params['param_range']['optimizer_range']),
                    'dropout'        : hp.uniform('dropout', params['param_range']['dropout_low'], params['param_range']['dropout_high']),
                    'batch_size'     : hp.choice('batch_size', params['param_range']['batch_range'])
        }

        space['attack_type']        = attack_type
        space['load_file_dir']      = load_file_dir

        space['dense_fix']          = params['dense_fix']
        space['activation_fix']     = params['activation_fix']
        space['filter_size']        = params['filter_size']
        space['padding']            = params['padding']
        space['loss']               = params['loss']
        space['epochs']             = params['epochs']
        space['verbose']            = params['verbose']
        space['shuffle']            = params['shuffle']
        space['metrics']            = params['metrics']

        space['max_evals']          = int(params['max_evals'])

        return space

    def _get_best_model(self, b_params, b_model, b_trial, x_test, y_test):
        best_params = b_params
        best_model  = b_model
        best_trial  = b_trial

        model_json  = best_model.to_json()
        data        = json.loads(model_json)

        loss, acc = best_model.evaluate(x_test, y_test)

        return best_params, best_model, acc

"""
Inference model
-----------------------------------
1. Load data
2. Convert data and preprocess
    2-1. Add temporay label column
    2-2. Convert action value to binary format
    2-3. Set model type accroding to attack name
    2-4. Payload preprocess
3. Retrun result dataset after inference process
"""
class known_model_inference():
    def __init__(self, gpu_dev=None):
        self.dls = data_load_save()
        self.kap = KnownAttackPreprocess()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev)

    def inf(self, load_file_dir, load_model_dir, load_model_wt_dir, load_file_attack_dir, params, attack_type, max_padd_size):
        # Load input data
        data = self.dls.get_data_df(load_file_dir, attack_type)

        # Convert data, preprocess
        df = self._data_conv(data, attack_type, load_file_attack_dir, params['action_value'])
        
        # Predict label
        gpu_context = gpu_context_enter(self.context, self.gpu_dev)
        df_inf = self._model_predict(df, load_model_dir, load_model_wt_dir, max_padd_size, params, attack_type)
        gpu_context_exit(gpu_context)
        
        return df_inf

    def _data_conv(self, data, attack_type, load_file_attack_dir, action_val):
        if data.empty or data is None:
            print("Data None") # Error log level

            return False
        else:
            # 1. Add temporay label
            data['label'] = -1
            data['label'] = data['label'].astype(int)
            
            # 2. Convert action value to binary format (0, 1)
            # data.loc[:, 'action_val'] = data['action'].apply(lambda x: self._convert_action_value(x, action_val))
            data.loc[:,'action_val'] = data['action'].apply(cm.convert_action_value)
            print(data['action_val'].value_counts())
            # df['action_val']  = df['Action'].apply(cm.convert_action_value)
            
            # 3. Set model type accroding to attack name
            data['attack_nm'] = data['attack_nm'].astype(str)
            
            attack_nm_pkl_file = load_file_attack_dir.format(attack_type)
            attack_nm = self._set_model_type(attack_nm_pkl_file)
            data['model_type'] = data['attack_nm'].apply(lambda x: attack_type if x.strip() in attack_nm else '')
            
            df_a = data[data['model_type'] == attack_type].copy()
            df_a = df_a.reset_index(drop=True)
            # print(df_a.shape) # for test
            
            # 4. Preprocess payload
            df_a['payload_ps'] = df_a['payload'].apply(lambda x: self.kap.preprocess_payload_str(attack_type, x))

            return df_a


    def _convert_action_value(self, action_name, action_value):
        if type(action_name) != str and math.isnan(action_name):
            action_name = 'etc' 
        return action_value[action_name]
    
    
    # Load related attack_nm list from pickle file
    def _set_model_type(self, attack_nm_pkl_file):        
        with open(attack_nm_pkl_file, 'rb') as f:
            attack_nm = pickle.load(f)
            
        return attack_nm
        
    
    def _model_predict(self, data, load_model, load_model_wt, max_padd_size, params, attack_type):
        inf_dt = time.strftime('%Y%m%d%H%M%S')

        load_model_file      = load_model.format(attack_type)
        load_model_wt_file   = load_model_wt.format(attack_type)

        # Load model
        load_model = self.dls.get_load_model_file(load_model_file, load_model_wt_file, attack_type)

        # Compile
        load_model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])

        data['predict'] = data['payload_ps'].apply(lambda x: load_model.predict(np.reshape(self.kap.asc_padding(x, max_padd_size, 0)[1:DIM_INPUT], (DIM_INPUT - 1, 1)).T))

        data['0'] = data['predict'].apply(lambda x: x[0][0])
        data['1'] = data['predict'].apply(lambda x: x[0][1])
        data['label_inf'] = data.apply(lambda x: 0 if x['0'] > x['1'] else 1, axis=1)
        data['prob_inf'] = data.apply(lambda x: x['0'] if x['0'] > x['1'] else x['1'], axis=1)
        
        data['inf_dt'] = inf_dt

        del data['label']
        
        return data


class data_load_save():
    def get_data_file(self, load_dir, attack_type):
        if attack_type is None:
            return False
        
        dataFile_dir = load_dir.format(attack_type)
        dataset = pd.read_csv(dataFile_dir).values
        # print(dataset[:5]) # for test

        train, test = train_test_split(dataset, test_size = 0.1, random_state = 1)
        y_train, x_train = np_utils.to_categorical(train[:, 0]), train[:, 1:]
        y_test, x_test = np_utils.to_categorical(test[:, 0]), test[:, 1:]

        return dataset, x_train, y_train, x_test, y_test


    def get_data_df(self, load_dir, attack_type):
        if attack_type is None:
            return False
        
        dataFile_dir = load_dir.format(attack_type)
        df = pd.read_csv(dataFile_dir, encoding='utf-8')

        # Delete payload empty value
        dataset = df[~df['payload'].isnull()]

        return dataset

    # if use n% split
    def get_data_file_split(self, load_dir, attack_type):
        if attack_type is None:
            return False
        
        dataFile_dir = load_dir.format(attack_type)
        dataset = pd.read_csv(dataFile_dir, header = None)

        '''if use 20% split data'''
        ratio = 0.2
        start_seq = 1

        size = int(len(dataset) * ratio)
        list_of_datas = [dataset.iloc[i : i + size, :] for i in range(0, len(dataset), size)]
        new_data = list_of_datas[start_seq].values

        train, test = train_test_split(new_data, test_size = 0.1, random_state = 1)
        y_train, x_train = np_utils.to_categorical(train[:, 0]), train[:, 1:]
        y_test, x_test = np_utils.to_categorical(test[:, 0]), test[:, 1:]

        dataset = new_data

        return dataset, x_train, y_train, x_test, y_test

    def get_save_df(self, data, save_dir, attack_type):
        file = save_dir.format(attack_type)
        if data.empty or data is None:
            print("Data None") # Error log level
            
            return False
        else:
            data.to_csv(file, index=False)
            
            return True

    def save_model_and_weight(self, model, save_model_dir, save_model_wt_dir, attack_type):
        save_model_file     = save_model_dir.format(attack_type)
        save_model_wt_file  = save_model_wt_dir.format(attack_type)
        if (model):
            # save model
            with open(save_model_file, "w") as json_file:
                json_file.write(model.to_json())
            
            # save weights
            model.save_weights(save_model_wt_file)
            
            # print('save model : {}'.format(save_model_file)) # for test
            # print('save model wt : {}'.format(save_model_wt_file)) # for test

            return True
        else:
            return False


    def get_load_model_file(self, model_dir, model_wt_dir, attack_type):
        model_file      = model_dir.format(attack_type)
        model_wt_file   = model_wt_dir.format(attack_type)

        # Load model
        json_file = open(model_file)
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)

        model.load_weights(model_wt_file)

        '''for test'''
        # print('loaded model architecture')
        # print(model.summary())
        # print('')
        # print('load model : {}'.format(model_file))
        # print('load model wt : {}'.format(model_wt_file))

        return model