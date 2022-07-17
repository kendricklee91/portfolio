from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import taxf_config as config
import taxf_utils as utils
import pandas as pd
import numpy as np
import joblib
import pickle
import os

logger = utils.get_logger(__name__)

class Model():
    def __init__(self, df:pd.DataFrame, target:str = 'CD_ACCOUNT'):
        self.data_path = config.DATA_PATH
        self.model_path = config.MODEL_PATH
        self.target = target
        
        self.data = df
        self.X, self.y = self._prepare_data()
        self.train_X, self.val_X, self.train_y, self.val_y = self._train_test_split()
        self.unique_class = list(self.y.unique())

        self.threshold_per_cls = {}
        self.result_dict = {}
        self.model = None

    def _prepare_data(self):
        logger.info('Prepare data')

        if self.target.lower() in ['account', 'cd_account']:
            X = self.data.drop(['CD_ACCOUNT', 'CD_DEDU'], axis = 1).astype(int)
            y = self.data.CD_ACCOUNT
        elif self.target.lower() in ['dedu', 'cd_dedu']:
            df = self.data.loc[(self.data.CD_DEDU != '') & (self.data.CD_DEDU != '2')]
            df = df.dropna(subset = ['CD_DEDU']).reset_index(drop = True)

            X = df.drop(['CD_ACCOUNT', 'CD_DEDU'], axis = 1).astype(int)
            y = df.CD_DEDU
        return (X, y)
    
    def _train_test_split(self):
        logger.info('Split train, validation set')

        while 1:
            train_X, val_X, train_y, val_y = train_test_split(self.X, self.y, test_size = config.TEST_SIZE, random_state = 4710)
            if (train_y.nunique() == self.y.nunique()):
                break
        return (train_X, val_X, train_y, val_y)
    
    def param_tuning(self):
        logger.info('Parameters tuning')

        depths = range(24, 40, 2)
        estimators = range(64, 128, 4)
        
        logger.info('Searching best hyper-parameters')
        self.result_dict = {}
        for depth in depths:
            for estimator in estimators:
                model = RandomForestClassifier(n_estimators = estimator, max_depth = depth, n_jobs = -1, verbose = 0, random_state = 42)
                model.fit(self.train_X, self.train_y)
                acc = accuracy_score(self.val_y, model.predict(self.val_X))
                self.result_dict[(depth, estimator)] = acc
    
    def get_best_params(self):
        depth, estimator = sorted(self.result_dict.items(), key = lambda x: x[1], reverse = True)[0][0]
        
        result_params = {}
        result_params['max_depth'] = depth
        result_params['n_estimators'] = estimator
        return result_params

    def train(self):
        if self.result_dict != {}:
            best_params = self.get_best_params()
            self.model = RandomForestClassifier(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], n_jobs = -1, verbose = 0, random_state = 42)
        else:
            # Training with default parameters
            self.model = RandomForestClassifier(n_estimators = 104, max_depth = 34, n_jobs = -1, verbose = 0, random_state = 42)
        
        self.model.fit(self.train_X, self.train_y)
        self._save_model()
    
    def _save_model(self):
        logger.info('Saving model')

        model_path = f"{self.model_path}/model_{self.target}_{config.WORD_FREQ}.pkl"
        if not os.path.exists(model_path):
            joblib.dump(self.model, model_path)
            acc = accuracy_score(self.val_y, self.model.predict(self.val_X))
            logger.info(f"Accuracy of model for validation set : {acc}")
        else:
            prev_model = joblib.load(model_path)
            prev_acc = accuracy_score(self.val_y, prev_model.predict(self.val_X))
            acc = accuracy_score(self.val_y, self.model.predict(self.val_X))
            logger.info(f"Accuracy of previous model : {prev_acc}")
            logger.info(f"Accuracy of current model : {acc}")

            if acc > prev_acc:
                logger.info("Previous model replaced by current model")
                joblib.dump(self.model, model_path)
            else:
                if os.path.exists(f"{config.DATA_PATH}/concat_in_prediction_backup.pkl"):
                    os.remove(f"{config.DATA_PATH}/concat_in_prediction.pkl")
                    os.rename(f"{config.DATA_PATH}/concat_in_prediction_backup.pkl", f"{config.DATA_PATH}/concat_in_prediction.pkl")
    
    def get_model(self):
        logger.info('Get model')

        model_path = f"{self.model_path}/model_{self.target}_{config.WORD_FREQ}.pkl"
        self.model = joblib.load(model_path)
    
    def pred_summarize(self):
        pred_val = self.model.predict_proba(self.val_X)
        pred_cls = list(map(lambda x: self.model.classes_[x], np.argmax(pred_val, axis = 1)))
        pred_prob = np.max(pred_val, axis = 1)
        real_val = self.val_y

        df = pd.DataFrame({'real_cls' : real_val, 'pred_cls' : pred_cls, 'pred_prob' : pred_prob})
        df.loc[:, 'correct'] = np.where(df.real_cls == df.pred_cls, 1, 0)
        df.loc[:, 'pred_prob'] = np.round(df.pred_proba, 1)
        return df
    
    def thresh_per_cls(self):
        summarize_df = self.pred_summarize()
        for cls in self.unique_class:
            try:
                threshold = summarize_df.groupby('real_cls').get_group(cls).groupby('pred_proba').correct.value_counts(normalize = True).reset_index(name = 'ratio').query(f"correct == 1 and ratio > {config.THRESHOLD}").pred_prob.values[0]
                self.threshold_per_cls[cls] = threshold
            except IndexError:
                self.threshold_per_cls[cls] = 1
        joblib.dump(self.threshold_per_cls, f"{self.data_path}/thresh_per_cls_{self.target}_{config.WORD_FREQ}.pkl")
    
