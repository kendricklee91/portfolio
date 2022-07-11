from ksv_model.model_payload import known_model, known_model_retrain, known_model_hopt, known_model_inference, data_load_save
import ksv_model.config.const as cst

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from scipy import stats

import pandas as pd
import unittest
import json
import os

class TestModel1st(unittest.TestCase):

    # Fixture
    def setUp(self):
        pass

    def tearDown(self):
        pass


    # Inference test of sql injection model
    @unittest.skip
    def test_sqli_inference(self):
        attack_type             = 'sqli'
        max_padd_size           = 1024

        load_file_dir           = os.path.join(cst.PATH_DATA, '{}_ps_3dayvalid_01.csv')
        load_file_attack_dir    = os.path.join(cst.PATH_DATA, 'attack_nm_{}.pkl')
        save_file_dir           = os.path.join(cst.PATH_DATA, '{}_inf_result_01.csv')

        load_model_dir          = os.path.join(cst.PATH_MODEL, '{}_train_01.json')
        load_model_wt_dir       = os.path.join(cst.PATH_MODEL, '{}_train_01.h5')

        # Model parameters
        model_params        = os.path.join(cst.PATH_CONFIG, 'model_payload_param.json')
        with open(model_params, encoding='utf-8') as json_file:
            params = json.load(json_file)
            
        km_inf = known_model_inference()
        result_data = km_inf.inf(load_file_dir, load_model_dir, load_model_wt_dir, load_file_attack_dir, params, attack_type, max_padd_size)

        # Save inference result as csv file
        dls     = data_load_save()
        result  = dls.get_save_df(result_data, save_file_dir, attack_type)
        # print(result) # for test

    # Inference test of sql injection model
    def test_ig_inference(self):
        load_file_dir           = os.path.join(cst.PATH_DATA, 'ig_ps_20190831_inference_01.csv')
        save_file_dir           = os.path.join(cst.PATH_DATA, 'ig_inf_result_01.csv')

        load_model_dir          = os.path.join(cst.PATH_MODEL, 'ig_train_01.json')
        load_model_wt_dir       = os.path.join(cst.PATH_MODEL, 'ig_train_01.h5')

        # Model parameters
        model_params        = os.path.join(cst.PATH_CONFIG, 'model_ig_param.json')
        with open(model_params, encoding='utf-8') as json_file:
            params = json.load(json_file)
            
        km_inf = ModelIG_inference()
        result_data = km_inf.inf(load_file_dir, load_model_dir, load_model_wt_dir, params)

        # Save inference result as csv file
        kap             = KnownAttackPreprocess2()
        result          = kap.get_save_df(result_data, save_file_dir)
        # print(result) # for test

if __name__ == "__main__":
    unittest.main()