from ksv_model.model_payload import known_model, known_model_retrain, known_model_hopt, known_model_inference, data_load_save
from ksv_model.model_ig import ModelIG, ig_data_load_save
import ksv_model.config.const as cst

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from scipy import stats

import pandas as pd
import unittest
import json
import os
# import pprint # for test


class TestModel1st(unittest.TestCase):

    # Fixture
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # normal train test of sql injection model
    def test_ig_train(self):

        
        load_file_dir       = os.path.join(cst.PATH_DATA, 'ig_ps_20190830_01.csv')
        save_model_dir      = os.path.join(cst.PATH_MODEL, 'ig_train_01.json')
        save_model_wt_dir   = os.path.join(cst.PATH_MODEL, 'ig_train_01.h5')
        
        # model parameters
        model_params         = os.path.join(cst.PATH_CONFIG, 'model_ig_param.json')
        with open(model_params, encoding='utf-8') as json_file:
            params = json.load(json_file)

        kmt = ModelIG()
        model, hist, loss, acc = kmt.create_model(load_file_dir, params)
        # print(acc) # for test
        
        # save model
        dls         = ig_data_load_save()
        save_result = dls.save_model_and_weight(model, save_model_dir, save_model_wt_dir)
    
    
if __name__ == "__main__":
    unittest.main()