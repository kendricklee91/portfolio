# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime


def preprocess(df, feature_mapping, userid_mean, quizcode_mean, mcode_mean, input_path, column_definitions):
    """
    DB에서 가져온 문항 데이터를 모델에서 추론 가능한 형태(ffm file)로 전처리

    Parameters
    ----------
    df: pandas.Dataframe
        문항 데이터
    feature_mapping: dict
        feature map
    userid_mean: pandas.Dataframe
        average userid mean from train
    quizcode_mean: pandas.Dataframe
        average quizcode mean from train
    mcode_mean: pandas.Dataframe
        average mcode mean from train
    input_path: str
        문항 데이터의 ffm 파일 경로
    column_definitions: dict

    Returns
    -------
    str
        문항 데이터의 ffm 파일 경로
    """
    assert len(df) > 0, 'Dataframe is empty'

    # the folders and build the input file name
    now = datetime.now()
    path = os.path.join(input_path, str(now.year), str(now.month), str(now.day), str(now.hour))
    os.makedirs(path, exist_ok=True)

    input_filepath = os.path.join(path, f'{uuid.uuid4()}-{now.timestamp()}')

    # column type fix
    for col in column_definitions:
        if col in ['userid_mean', 'quizcode_mean', 'mcode_mean']:
            continue

        if col in ['ptype_cd', 'content_cd', 'cognitive_cd']:
            df[col] = df[col].astype(float)

        _type = column_definitions[col]['dtype']
        df[col] = df[col].astype(_type)

    df.UserID = df.UserID.apply(lambda x: x.lower())  # convert userid to lowercase

    df = _merge_means(df, userid_mean, quizcode_mean, mcode_mean)
    df = _process_nulls(df, userid_mean, quizcode_mean, mcode_mean)
    df = df[list(column_definitions)]  # fix column order
    _write_ffm_file(input_filepath, df.values, column_definitions, feature_mapping)  # write input file for inference

    return input_filepath


def _merge_means(df, userid_mean, quizcode_mean, mcode_mean):
    """
    User ID mean values, quizCode mean values, mCode mean values를 문항 데이터에 merging

    Parameters
    ----------
    df: pandas.Dataframe
        문항 데이터
    userid_mean: pandas.Dataframe
        average userid mean from train
    quizcode_mean: pandas.Dataframe
        average quizcode mean from train
    mcode_mean: pandas.Dataframe
        average mcode mean from train

    Returns
    -------
    pandas.Dataframe
        mean values가 merging된 문항 데이터
    """
    df = pd.merge(df, userid_mean, how='left', on='UserID')
    df = pd.merge(df, quizcode_mean.astype('object'), how='left', on='QuizCode')
    df = pd.merge(df, mcode_mean, how='left', on='mCode')
    return df


def _process_nulls(df, userid_mean, quizcode_mean, mcode_mean):
    """
    User ID, quizCode, mCode의 mean value 컬럼의 N/A 값을 전체 평균값으로 변경

    Parameters
    ----------
    df: pandas.Dataframe
        문항 데이터
    userid_mean: pandas.Dataframe
        average userid mean from train
    quizcode_mean: pandas.Dataframe
        average quizcode mean from train
    mcode_mean: pandas.Dataframe
        average mcode mean from train

    Returns
    -------
    pandas.Dataframe
        mean value 컬럼의 N/A 값이 처리된 문항 데이터
    """

    # calculate overall average
    avg_userid_mean = userid_mean.userid_mean.mean()
    avg_quizcode_mean = quizcode_mean.quizcode_mean.mean()
    avg_mcode_mean = mcode_mean.mcode_mean.mean()

    _tmp = {'userid_mean': avg_userid_mean, 'quizcode_mean': avg_quizcode_mean, 'mcode_mean': avg_mcode_mean}

    for col, avg in _tmp.items():
        df[col] = df[col].fillna(avg).replace('None', avg)

    return df


def _write_ffm_file(output_filepath, arr, column_definitions, feature_mapping):
    """
    문항 데이터를 모델에서 추론 가능한 형태(ffm file)로 만들어 파일에 쓰기

    Parameters
    ----------
    output_filepath: str
        생성할 ffm 파일 경로
    arr: list
        문항 데이터의 dataframe value list
    column_definitions: dict
        컬럼 설정 정보
    feature_mapping
        feature map

    Returns
    -------
    None
    """
    with open(output_filepath, 'w') as f:
        rc = 0  # keep track of current row
        for row in arr:
            line = []  # output line
            for name, value in zip(column_definitions, row):
                col_id = column_definitions[name]['id']
                col_type = column_definitions[name]['type']

                if value is None or (isinstance(value, float) and np.isnan(value) or pd.isna(value)):
                    # null value skip
                    continue

                if col_type == 'label':
                    line.insert(0, str(value))  # Label should be inserted at the front of the input line
                elif col_type == 'con':
                    feature_index = feature_mapping[name]['features']
                    line.append(f'{col_id}:{feature_index}:{value:.5f}')
                elif col_type == 'cat':
                    try:
                        feature_index = feature_mapping[name]['features'][value]
                        line.append(f'{col_id}:{feature_index}:1')
                    except:
                        pass
                else:
                    raise ValueError(f'Column type [{col_type}] is not valid. Must be one of [label, con, cat]')

            if len(line) == 0:
                raise ValueError(f'It appears all values in the row arr[{rc}, :] are None or numpy.nan')

            line = ' '.join(line)
            f.write(line)
            f.write('\n')
            rc += 1
