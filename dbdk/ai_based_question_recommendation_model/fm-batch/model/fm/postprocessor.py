# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import pickle


def postprocess(model_path, feature_mapping, userid_mean, quizcode_mean, mcode_mean):
    """
    feature map, User ID mean values, quizCode mean values, mCode mean values를 파일로 저장

    Parameters
    ----------
    model_path: str
        모델 디렉토리 경로
    feature_mapping: dict
        feature map
    userid_mean: pandas.Dataframe
        average userid mean from train
    quizcode_mean: pandas.Dataframe
        average quizcode mean from train
    mcode_mean: pandas.Dataframe
        average mcode mean from train

    Returns
    -------
    None
    """
    os.makedirs(model_path, exist_ok=True)

    # mapping dict info save
    with open(os.path.join(model_path, 'feature_mapping.pkl'), 'wb') as mapping_file:
        pickle.dump(feature_mapping, mapping_file)

    # mean values save to pickle
    userid_mean.to_pickle(os.path.join(model_path, 'userid_mean.pkl'))
    quizcode_mean.to_pickle(os.path.join(model_path, 'quizcode_mean.pkl'))
    mcode_mean.to_pickle(os.path.join(model_path, 'mcode_mean.pkl'))
