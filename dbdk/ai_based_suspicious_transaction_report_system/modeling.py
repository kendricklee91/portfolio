import packages as pkgs
import config as cfg
import utils
import shap

logger = utils.set_logger(__name__)

def imb_data_process(x_train, y_train):
    logger.info('데이터 불균형 처리 시작')
    #smnc = pkgs.SMOTENC(categorical_features = np.where(X_train.columns.str.contains('Yn'))[0], random_state = 34, n_jobs = -1)
    
    sampler = pkgs.RandomOverSampler(random_state = cfg.RANDOM_SEED)
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    logger.info('데이터 불균형 처리 종료')
    return x_resampled, y_resampled
    
def data_split(total_data, case):
    logger.info('데이터 분리 시작')
    total_data = total_data.loc[total_data['anw_period'] >= 0].reset_index(drop = True) # anw_period = -1 제거
    total_data.columns = [i.strip(' ') for i in total_data.column.values] # Column 내 띄어쓰기 제거
    total_data['label_dt'] = total_data['tr_dt'] # total_data 내 label_dt 컬럼 생성
    
    tsd = cfg.TRAIN_START_DATE
    ted = cfg.TRAIN_END_DATE
    
    if 1 == case:
        if cfg.TRAIN_VALIDATION_SPLIT_BY_PERIOD:
            vsd = cfg.VALIDATION_START_DATE
        else:
            test_size = cfg.TRAIN_IN_VALID_SIZE
            random_seed = cfg.RANDOM_SEED
    
        # train : label_dt <= 2021-05-31
        x_train_, y_train = total_data[total_data['label_dt'] <= ted], total_data.loc[total_data['label_dt'] == ted, 'suspicious']

        # test : label_dt > 2021-05-31
        x_test_, y_test = total_data[total_data['label_dt'] > ted], total_data.loc[total_data['label_dt'] > ted, 'suspicious']

        if cfg.TRAIN_VALIDATION_SPLIT_BY_PERIOD:
            train_idx = x_train_[x_train_['label_dt'] < varlidation_start_date].sample(frac = 1, random_state = RANDOM_SEED).index

            x_valid_, y_valid = x_train_[x_train_['label_dt'] >= vsd], x_train_[x_train_['label_dt'] >= vsd]['suspicious']
            x_train_, y_train = x_train_.loc[train_idx], y_train.loc[train_idx]
        else:
            x_train_, x_valid_, y_train, y_valid = pkgs.train_test_split(x_train_, x_train_['suspicious'], test_size = test_size, shuffle = True,
                                                                         stratify = x_train_['suspicious'], random_state = cfg.RANDOM_SEED)    
        x_train = x_train_.drop(['cusno', 'suspicious', 'label_dt', 'tr_dt', 'bsn_dsc'], axis = 1)
        x_valid = x_valid_.drop(['cusno', 'suspicious', 'label_dt', 'tr_dt', 'bsn_dsc'], axis = 1)
        x_test = x_test_.drop(['cusno', 'suspicious', 'label_dt', 'tr_dt', 'bsn_dsc'], axis = 1)
        logger.info('데이터 분리 종료')
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        x_train_, x_test_, y_train, y_test = pkgs.train_test_split(train_data.dorp(['suspicious'], axis = 1),
                                                                   train_data['suspicious'], test_size = 0.2, shuffle = True,
                                                                   stratify = train_data['suspicious'], random_state = cfg.RANDOM_SEED)
        x_train = x_train_.drop(['cusno', 'label_dt', 'tr_dt', 'bsn_dsc'], axis = 1)
        x_test = x_test_.drop(['cusno', 'label_dt', 'tr_dt', 'bsn_dsc'], axis = 1)
        logger.info('데이터 분리 종료')
        return x_train, _, x_test, y_train, _, y_test
    
def model(total_data, case):
    logger.info('모델 학습 시작')
    x_train, x_valid, x_test, y_train, y_valid, y_test = data_split(total_data, case) # 데이터 분리 
    x_resampled, y_resampled = imb_data_process(x_train, y_train) # 데이터 불균형 처리
        
    if 1 == case:
        # 하이퍼 파라미터 튜닝
        if cfg.HYPERPARAMETER_OPT:
            start = pkgs.time.time()

            xgb_cv_partial = pkgs.partial(utils.xgb_cv, X_tmp = x_resampld, y_tmp = y_resampled, X_val_tmp = x_valid, y_val_tmp = y_valid)
            bo = pkgs.BayesianOptimization(f = xgb_cv_partial, pbounds = cfg.PBOUNDS, verbose = 2, random_state = 7)
            bo.maximize(init_points = 1, n_iter = 5)

            end = pkgs.time.tim()        
            print(f"Running time : {end - start}s")

            best_param = bo.max['params']
            print(best_param)
        else:
            pass
    
        # 하이퍼 파라미터 튜닝 후 모델 학습에 입력할 파라미터 값
        model = pkgs.XGBClassifier(
            max_depth        = int(best_param['max_depth']),
            learning_rate    = round(best_param['learning_rate'], 3),
            n_estimators     = int(best_param['n_estimators']),
            gamma            = round(best_param['gamma'], 3),
            min_child_weight = int(best_param['min_child_weight']),
            subsample        = round(best_param['subsample'], 3),
            colsample_bytree = round(best_param['colsample_bytree'], 3),
            alph             = round(best_param['alpha'], 3),
            reg_alpha        = round(best_param['reg_alpha'], 3),
            reg_lambda       = round(best_param['reg_lambda'], 3),
            n_jobs = 4, random_state = 34
        )
        model.fit(X_resampled, y_resampled, eval_set = [(X_valid, y_valid)], eval_metric = ['logloss'], early_stopping_rounds = 100, verbose = 100)
        logger.info('모델 학습 종료')        
    else:
        model = pkgs.XGBClassifier(n_estimators = 1000, n_jobs = 4, random_state = 34)
        model.fit(X_resampled, y_resampled, eval_metric = 'logloss')
        logger.info('모델 학습 종료')
    
    logger.info('모델 검증 시작')
    origin_label = utils.create_label(cfg.DTC_START_DATE, cfg.DTC_END_DATE, is_train = True, bsn_dsc = '05', cus_tpc = ['1', '2', '3'])
    x_test_ = x_test_.merge(origin_label[['cusno', 'tr_dt', 'dtc_dt']])
    
    result = x_test_
    result['predict'] = model.predict(x_test)
    result[['0_proba', '1_proba']] = model.predict_proba(x_test)
    result = pd.concat([result[cfg.RESULT_META_COLS], result.drop([cfg.RESULT_META_COLS], axis = 1)], axis = 1).reset_index(drop = True)
    logger.info('모델 검증 종료')

    # Performances of model
    confusion, accuracy, precision, recall, f1, auc = utils.get_clf_eval(y_test, result['predict'])
                                                                         
    metric_df = pkgs.pd.DataFrame({0 : {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall, 'f1' : f1, 'auc' : auc}}).T
    metric_df[['FF', 'FP', 'FN', 'TP']] = confusion.flatten()
    
    confusion_df = pkgs.pd.DataFrame(confusion)
    
    logger.info('모델의 결과 해석')
    plt.rcParams['figure.figsize'] = (10, 14)
    pkgs.plot_importance(model, max_num_feature = 80) # plot_importance(model, max_num_features = 80, importance_type = 'gain')
    
    shapely_top5_result = shap.shapely_top5_result(model, x_test, result)
    return metric_df, confusion_df, shapely_top5_result