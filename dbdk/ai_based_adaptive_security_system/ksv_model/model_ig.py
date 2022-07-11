import os, math 
import pandas as pd
import numpy as np

import tensorflow  as tf
import keras.backend.tensorflow_backend as K
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Activation, Dense, Dropout

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time, json, csv, os, sys, pickle
import ksv_model.config.gpu_dev as gd
import ksv_model.config.const as cst
from ksv_model.post_process import task_post_process_ig_model_inf, task_post_process_auto_block


def gpu_context_enter(context, gpu_dev):
    context = K.tf.device(gpu_dev)
    context.__enter__()

    return context

def gpu_context_exit(context):
    context.__exit__(None, None, None)


# 초기학습 모델 
class ModelIG:
    def __init__(self, gpu_dev=None):

        self.dls = ig_data_load_save()
        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev) 

    def create_model(self, load_file_dir, params):
        # with 구문 대신 사용
        gpu_context = gpu_context_enter(self.context, self.gpu_dev)


        #############
        # Data Load #
        #############
        x_train, y_train, x_test, y_test = self.dls.get_data_file(load_file_dir)



        #############
        # Model #
        #############
        model = Sequential()
        model.add(Dense(params["dense_layer_1"], input_dim=x_train.shape[1]) )
        
        model.add(Activation(params["activation_1"]))
        model.add(Dropout( params["dropout_1"]))
                
        model.add(Dense(params["dense_layer_2"] ))
        model.add(Activation( params["activation_2"]))
        model.add(Dropout( params["dropout_2"]))        
        
        model.add(Dense(params["dense_layer_3"]))
        model.add(Activation( params["activation_3"]))
        model.add(Dropout( params["dropout_3"]))    

        model.add(Dense(params["dense_fix"], activation=params["activation_fix"]))
        
        model.compile(loss = params['loss'], optimizer = params["optimizer"], metrics = [params['metrics']])

        hist = model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params['batch_size'], verbose = params['verbose'],
                         validation_data = (x_test, y_test), shuffle = params['shuffle'])

        gpu_context_exit(gpu_context)

        eval_loss, eval_acc = model.evaluate(x_test, y_test)

        # print('loss :', eval_loss, ', acc :', eval_acc) # for test
        return model, hist, eval_loss, eval_acc
   


# 정보수집 모델 재학습
class ModelIG_retrain:
    def __init__(self, gpu_dev=None):
        self.dls = ig_data_load_save()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev)

    def retrain_model(self, params):
        # load data
        x_train, y_train, x_test, y_test = self.dls.get_data_file(params['load_file_dir'])

        gpu_context = gpu_context_enter(self.context, self.gpu_dev)

        # load model
        load_model = self.dls.get_load_model_file(params['load_model_dir'], params['load_model_wt_dir'])
        
        # Training 
        load_model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])
        hist = load_model.fit(x_train, y_train, batch_size = params['batch_size'], epochs = params['epochs'],
                            verbose = params['verbose'], validation_data = (x_test, y_test), shuffle = params['shuffle'])

        gpu_context_exit(gpu_context)

        validation_acc = np.max(hist.history['val_acc'])

        return {'loss' : -validation_acc, 'status' : STATUS_OK, 'model' : load_model, 'x_test' : x_test, 'y_test' : y_test}

    def optimize(self, load_file_dir, load_model_dir, load_model_wt_dir, params):
        space       = self._get_space_retrain(load_file_dir, load_model_dir, load_model_wt_dir, params)

        trials      = Trials()
        best_run    = fmin(self.retrain_model, space, algo = tpe.suggest, max_evals = space['max_evals'], trials = trials)

        best_params = space_eval(space, best_run)
        best_model  = trials.best_trial['result']['model']
        best_trial  = trials.best_trial

        x_test = best_trial['result']['x_test']
        y_test = best_trial['result']['y_test']

        return self._get_best_model(best_params, best_model, best_trial, x_test, y_test)

    def _get_space_retrain(self, load_file_dir, load_model_dir, load_model_wt_dir, params):
        # Set parameter search bounds
        space = {
            'optimizer'      : hp.choice('optimizer', params['param_range']['optimizer_range']),
            'batch_size'     : hp.choice('batch_size', params['param_range']['batch_range'])
        }

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
# 정보수집 모델 최적화 학습
class ModelIG_hopt:
    def __init__(self, gpu_dev=None):
        self.dls = ig_data_load_save()

        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev)

    def create_model(self, params):
        # load data
        x_train, y_train, x_test, y_test = self.dls.get_data_file(params['load_file_dir'])

        gpu_context = gpu_context_enter(self.context, self.gpu_dev)

        model = Sequential()
        model.add(Dense(params["dense_size_1"], input_dim=x_train.shape[1], name = 'dense_size_1' ) )
        
        model.add(Activation(params["activation_1"], name = 'activation_1'))
        model.add(Dropout( params["dropout"], name = 'dropout_1' ))
                
        model.add(Dense(params["dense_size_2"], name = 'dense_size_2' ))
        model.add(Activation( params["activation_2"], name = 'activation_2' ))
        model.add(Dropout( params["dropout"], name = 'dropout_2' ))        

        model.add(Dense(params["dense_size_3"], name = 'dense_size_3' ))
        model.add(Activation( params["activation_3"], name = 'activation_3'  ))
        model.add(Dropout( params["dropout"], name = 'dropout_3' ))    

        #model.add(Dense(params["dense_size_4"], name = 'activation='sigmoid', name='dense_size_4'))
        model.add(Dense(params["dense_fix"], activation=params["activation_fix"]))

        model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])

        hist = model.fit(x_train, y_train, batch_size = params['batch_size'], epochs = params['epochs'],
            verbose = params['verbose'], validation_data = (x_test, y_test), shuffle = params['shuffle'])
                                    
        gpu_context_exit(gpu_context)
        
        validation_acc = np.max(hist.history['val_acc'])
        print('Best validation acc of epoch : ', validation_acc) # for test

        return {'loss' : -validation_acc, 'status' : STATUS_OK, 'model' : model, 'x_test' : x_test, 'y_test' : y_test}
    
    def optimize(self, load_file_dir, load_model_dir, load_model_wt_dir, params):
        space       = self._get_space_hopt(load_file_dir, load_model_dir, load_model_wt_dir, params)

        trials      = Trials()
        best_run    = fmin(self.create_model, space, algo = tpe.suggest, max_evals = space['max_evals'], trials = trials)

        best_params = space_eval(space, best_run)
        best_model  = trials.best_trial['result']['model']
        best_trial  = trials.best_trial

        x_test = best_trial['result']['x_test']
        y_test = best_trial['result']['y_test']

        return self._get_best_model(best_params, best_model, best_trial, x_test, y_test)

    def _get_space_hopt(self, load_file_dir, load_model_dir, load_model_wt_dir, params):
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
                    'dense_size_3'   : hp.choice('dense_size_3', params['param_range']['dense_range']),
                    'activation_1'   : hp.choice('activation_1', params['param_range']['activation_range']),
                    'activation_2'   : hp.choice('activation_2', params['param_range']['activation_range']),
                    'activation_3'   : hp.choice('activation_3', params['param_range']['activation_range']),
                    'optimizer'      : hp.choice('optimizer', params['param_range']['optimizer_range']),
                    'dropout'        : hp.uniform('dropout', params['param_range']['dropout_low'], params['param_range']['dropout_high']),
                    'batch_size'     : hp.choice('batch_size', params['param_range']['batch_range'])
        }

        space['load_file_dir']      = load_file_dir

        space['dense_fix']          = params['dense_fix']
        space['activation_fix']     = params['activation_fix']
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


# 정보수집 추론
class ModelIG_inference():
    def __init__(self, gpu_dev=None):

        self.dls = ig_data_load_save()
        gd.set_dev_env()
        gd.set_gpu_mem()

        self.context = None
        self.gpu_dev = gd.set_gpu_dev(gpu_dev) 


    def inf(self, load_file_dir, load_model_dir, load_model_wt_dir, params):
        # Load input data
        df = self.dls.get_data_inf_file(load_file_dir)
        
        # Predict label
        gpu_context = gpu_context_enter(self.context, self.gpu_dev)
        df_inf = self._model_predict(df, load_model_dir, load_model_wt_dir, params)
        gpu_context_exit(gpu_context)
        
        print (df_inf.columns)
        # autoblock 
        # task_post_process_auto_block(df, model_type, th_auto):
        df_inf['action_val'] = df_inf['action_val'].apply(lambda x : 1 if x == 255 else 0)

        rt = task_post_process_auto_block(df_inf, "ig", False)
        rt.to_csv(os.path.join(cst.PATH_DATA,"ig_ps_post_auto_01.csv"))
        
        
        # 후처리를 위한 데이터 저장
        # task_post_process_ig_model_inf(df)

        rt2 = task_post_process_ig_model_inf(df_inf)
        rt2.to_csv(os.path.join(cst.PATH_DATA,"ig_ps_post_data_01.csv"))

        df_inf['key_inf']  = ""
        df_inf['key_ps']   = ""
        df_inf['model_id'] = "ig"

        return df_inf[['key_inf','key_ps','model_id','label_inf','prob_inf','inf_dt']]

        
    
    def _model_predict(self, data, load_model, load_model_wt, params):
        extract_columns = ["src_ip_a", "src_ip_b", "src_ip_c", "src_ip_d", "src_port", 'dstn_port', 'risk', 
                           'dstn_ip_a', 'dstn_ip_b', 'dstn_ip_c', 'dstn_ip_d',
                           'pkt_cnt', 'pkt_size', "vul_attack","count", "action_val","domestic","private",
                           'time_diff', 'diff_std', 'prtc_udp', 'prtc_tcp', 'prtc_ip', 'prtc_icmp',
                           "overseas","unique_attack_cnt","unique_src_port_cnt","unique_dstn_ip_cnt",
                           "packet_size_mean", "time_diff_sec","attack_intv","attack_per_sec"]

        inf_dt = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Load model
        load_model = self.dls.get_load_model_file(load_model, load_model_wt)

        # Compile
        load_model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = [params['metrics']])

        
        data['label_inf'] = load_model.predict_classes(data[extract_columns])
        # 라벨값이 1일 확률 
        data['prob_1']  = load_model.predict(data[extract_columns])

        # 라벨값이 0일 확률 
        data['prob_0']  = 1 - data['prob_1']

        # 각 라벨값이 나올 확률값 
        data['prob_inf'] = data[['label_inf', 'prob_0', 'prob_1']].apply(lambda x : x[1] if(x[0] == 0) else x[2], axis=1)


        data.drop(['prob_0', 'prob_1'], axis=1, inplace=True)
        
        data['inf_dt'] = inf_dt
        
        
        return data


# Data 및 Model 로드 및 세이브 
class ig_data_load_save():

    def __init__(self):
        pass

    def get_data_file(self, load_dir, test_size_ = 0.33):
        
        data = pd.read_csv(load_dir)
        extract_columns = ["src_ip_a", "src_ip_b", "src_ip_c", "src_ip_d", "src_port", 'dstn_port', 'risk', 
                           'dstn_ip_a', 'dstn_ip_b', 'dstn_ip_c', 'dstn_ip_d',
                           'pkt_cnt', 'pkt_size', "vul_attack","count", "action_val","domestic","private",
                           'time_diff', 'diff_std', 'prtc_udp', 'prtc_tcp', 'prtc_ip', 'prtc_icmp',
                           "overseas","unique_attack_cnt","unique_src_port_cnt","unique_dstn_ip_cnt",
                           "packet_size_mean", "time_diff_sec","attack_intv","attack_per_sec"]

        X_tmp = data[extract_columns]
        Y_tmp = data[['label']]

        x_train, x_test, y_train, y_test = train_test_split(X_tmp, Y_tmp, test_size=test_size_)

        
        return x_train, y_train, x_test, y_test

    def get_data_inf_file(self, load_dir, test_size_ = 0.33):
        
        data = pd.read_csv(load_dir)

        X_tmp = data.drop(['label'], axis=1)
        
        return X_tmp


    def save_model_and_weight(self, model, save_model_dir, save_model_wt_dir):
        if (model):
            # save model
            with open(save_model_dir, "w") as json_file:
                json_file.write(model.to_json())
            
            # save weights
            model.save_weights(save_model_wt_dir)
            
            # print('save model : {}'.format(save_model_file)) # for test
            # print('save model wt : {}'.format(save_model_wt_file)) # for test

            return True
        else:
            return False


    def get_load_model_file(self, model_dir, model_wt_dir):
        # Load model
        json_file = open(model_dir)
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)

        model.load_weights(model_wt_dir)

        '''for test'''
        # print('loaded model architecture')
        # print(model.summary())
        # print('')
        # print('load model : {}'.format(model_file))
        # print('load model wt : {}'.format(model_wt_file))

        return model