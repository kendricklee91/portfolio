# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import pandas as pd
from utility import deserialize_file


def load_data(params, model_path, ai_rules, db_connector):
    """
    FM 추론을 위한 DB 문항 데이터, feature map, UserID mean values, quizCode mean values, mCode mean values 가져오기

    Parameters
    -------------------
    params: dict
        API 요청 파라메터
    model_path: str
        모델 디렉토리 경로
    ai_rules: dict
        DB의 인공지능 룰셋
    db_connector: DBConnector

    Returns
    -------
    pandas.Dataframe
        DB의 문항 데이터
    dict
        feature map
    pandas.Dataframe
        UserID mean values
    pandas.Dataframe
        quizCode mean values
    pandas.Dataframe
        mCode mean values
    """
    query_name = ai_rules['QUERY_NAME_TO_DATALOAD']
    columns = ai_rules['COLUMNS_TO_DATALOAD']
    df = _read_df(db_connector, query_name, params, columns)

    feature_mapping, userid_mean, quizcode_mean, mcode_mean = _get_maps(model_path)

    return df, feature_mapping, userid_mean, quizcode_mean, mcode_mean


def _read_df(db_connector, query_name, params, columns):
    """
    DB 문항 데이터를 데이터 프레임으로 가져오기

    Parameters
    ----------
    db_connector: DBConnector
    query_name: str
    params: dict
    columns: list

    Returns
    -------
    pandas.Dataframe
        문항 데이터
    """
    with db_connector as connected:
        query = connected.get_query(query_name, 'read', params)
        df = pd.read_sql(query, connected.conn, columns=columns)

    return df


def _get_maps(path):
    """
    feature map, UserID mean values, quizCode mean values, mCode mean values를 파일에서 가져오기

    Parameters
    ----------
    path: str
        모델 디렉토리 경로

    Returns
    -------
    dict
        feature map
    pandas.Dataframe
        UserID mean values
    pandas.Dataframe
        quizCode mean values
    pandas.Dataframe
        mCode mean values
    """
    feature_mapping = deserialize_file(os.path.join(path, 'feature_mapping.pkl'))
    userid_mean = deserialize_file(os.path.join(path, 'userid_mean.pkl'))
    quizcode_mean = deserialize_file(os.path.join(path, 'quizcode_mean.pkl'))
    mcode_mean = deserialize_file(os.path.join(path, 'mcode_mean.pkl'))

    return feature_mapping, userid_mean, quizcode_mean, mcode_mean
