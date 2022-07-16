# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import xlearn


def train(model_path, train_data_path, test_data_path, model_params):
    """
    FM 모델 트레이닝

    Parameters
    ----------
    model_path: str
        모델 디렉토리 경로
    train_data_path: str
        train data의 ffm 파일 경로
    test_data_path: str
        test data의 ffm 파일 경로
    model_params: dict
        hyper parameters

    Returns
    -------
    float
        AUC
    """
    os.makedirs(model_path, exist_ok=True)
    model_file_path = os.path.join(model_path, 'model_dm.out')

    model = xlearn.create_ffm()
    model.setTrain(train_data_path)
    model.setValidate(test_data_path)
    model.setSigmoid()
    model.setQuiet()
    model.fit(model_params, model_file_path)

    model.setTest(test_data_path)
    pred = model.predict(model_file_path)

    true = []
    with open(test_data_path) as f:
        for line in f:
            line = line.split()
            answer = line[0]
            true.append(answer)

    auc = _calc_auc(true, pred)

    return auc


def _calc_auc(true, pred):
    """
    AUC 계산

    Parameters
    ----------
    true: list
    pred: ndarray

    Returns
    -------
    float
        AUC
    """
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(true, pred)
    return auc
