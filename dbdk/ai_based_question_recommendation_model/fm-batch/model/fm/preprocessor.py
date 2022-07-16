# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
import messages


def preprocess(df, input_path, column_definitions):
    """
    DB에서 가져온 문항 데이터를 모델에서 추론 가능한 형태(ffm file)로 전처리

    Parameters
    ----------
    df: pandas.Dataframe
        문항 데이터
    input_path: str
        문항 데이터의 ffm 파일 경로
    column_definitions: dict

    Returns
    -------
    str
        train data ffm 파일 경로
    str
        test data ffm 파일 경로
    dict
        feature map
    pandas.Dataframe
        average userid mean from train
    pandas.Dataframe
        average quizcode mean from train
    pandas.Dataframe
        average mcode mean from train
    """
    assert len(df) > 0, messages.MSG_DF_EMPTY

    now = datetime.now()
    path = os.path.join(input_path, str(now.year), str(now.month), str(now.day), str(now.hour))
    os.makedirs(path, exist_ok=True)

    train_data_filepath = os.path.join(path, f'train-{uuid.uuid4()}-{now.timestamp()}')
    test_data_filepath = os.path.join(path, f'test-{uuid.uuid4()}-{now.timestamp()}')

    train_df, test_df = _train_test_split_dataframe(df)

    # column type fix
    for col in column_definitions:
        if col in ['userid_mean', 'quizcode_mean', 'mcode_mean']:
            continue

        if col in ['ptype_cd', 'content_cd', 'cognitive_cd']:
            train_df[col] = train_df[col].astype(float)
            test_df[col] = test_df[col].astype(float)

        _type = column_definitions[col]['dtype']
        train_df[col] = train_df[col].astype(_type)
        test_df[col] = test_df[col].astype(_type)

    # convert userid to lowercase
    train_df.UserID = train_df.UserID.apply(lambda x: x.lower())
    test_df.UserID = test_df.UserID.apply(lambda x: x.lower())

    train_df, test_df, userid_mean, quizcode_mean, mcode_mean = _generate_means(train_df, test_df)
    train_df, test_df = _process_nulls(train_df, test_df, userid_mean, quizcode_mean, mcode_mean)

    # fix column order
    train_df = train_df[list(column_definitions)]
    test_df = test_df[list(column_definitions)]

    feature_mapping = _create_feature_map(train_df, column_definitions)
    _write_ffm_file(train_data_filepath, train_df.values, column_definitions, feature_mapping)
    _write_ffm_file(test_data_filepath, test_df.values, column_definitions, feature_mapping)

    return train_data_filepath, test_data_filepath, feature_mapping, userid_mean, quizcode_mean, mcode_mean


def _train_test_split_dataframe(df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.1)
    return train, test


def _generate_means(train_df, test_df):
    """
    User ID mean values, quizCode mean values, mCode mean values를 구하여 문항 데이터에 merging

    Parameters
    ----------
    train_df: pandas.Dataframe
        train data
    test_df: pandas.Dataframe
        test data

    Returns
    -------
    pandas.Dataframe
        mean values가 merging된 train data
    pandas.Dataframe
        mean values가 merging된 test data
    pandas.Dataframe
        average userid mean from train
    pandas.Dataframe
        average quizcode mean from train
    pandas.Dataframe
        average mcode mean from train
    """
    # calculate averages
    userid_mean = train_df.groupby('UserID').agg({'Correct': ['mean']}).reset_index()
    quizcode_mean = train_df.groupby('QuizCode').agg({'Correct': ['mean']}).reset_index()
    mcode_mean = train_df.groupby('mCode').agg({'Correct': ['mean']}).reset_index()

    # reset columns for merging
    userid_mean.columns = ['UserID', 'userid_mean']
    quizcode_mean.columns = ['QuizCode', 'quizcode_mean']
    mcode_mean.columns = ['mCode', 'mcode_mean']

    # merge back into original dataframes
    train_df = pd.merge(train_df, userid_mean, how='left', on='UserID')
    train_df = pd.merge(train_df, quizcode_mean, how='left', on='QuizCode')
    train_df = pd.merge(train_df, mcode_mean, how='left', on='mCode')

    test_df = pd.merge(test_df, userid_mean, how='left', on='UserID')
    test_df = pd.merge(test_df, quizcode_mean, how='left', on='QuizCode')
    test_df = pd.merge(test_df, mcode_mean, how='left', on='mCode')

    return train_df, test_df, userid_mean, quizcode_mean, mcode_mean


def _process_nulls(train_df, test_df, userid_mean, quizcode_mean, mcode_mean):
    """
    User ID, quizCode, mCode의 mean value 컬럼의 N/A 값을 전체 평균값으로 변경

    Parameters
    ----------
    train_df: pandas.Dataframe
        train data
    test_df: pandas.Dataframe
        test data
    userid_mean: pandas.Dataframe
        average userid mean from train
    quizcode_mean: pandas.Dataframe
        average quizcode mean from train
    mcode_mean: pandas.Dataframe
        average mcode mean from train

    Returns
    -------
    pandas.Dataframe
        train data
    pandas.Dataframe
        test data
    """

    # calculate overall average
    avg_userid_mean = userid_mean.userid_mean.mean()
    avg_quizcode_mean = quizcode_mean.quizcode_mean.mean()
    avg_mcode_mean = mcode_mean.mcode_mean.mean()

    _tmp = {'userid_mean': avg_userid_mean, 'quizcode_mean': avg_quizcode_mean, 'mcode_mean': avg_mcode_mean}

    for col, avg in _tmp.items():
        train_df[col] = train_df[col].fillna(avg).replace('None', avg)
        test_df[col] = test_df[col].fillna(avg).replace('None', avg)

    return train_df, test_df


def _create_feature_map(df, column_definitions):
    """
    feature map 생성

    Parameters
    ----------
    df: pandas.Dataframe
        feature map을 생성할 데이터
    column_definitions: dict
        컬럼 설정 정보

    Returns
    -------
    dict
        feature map
    """
    mapping_dict = {}
    ft_ix = 0
    for name, col in column_definitions.items():
        col_id = col['id']
        col_type = col['type']

        if col_type == 'label':
            # We do not map the label
            continue

        if col_type == 'con':
            mapping_dict[name] = {'id': col_id, 'features': ft_ix}
            ft_ix += 1
            continue

        if col_type == 'cat':
            unique = list(df[name].dropna().unique())
            mapping_dict[name] = {'id': col_id, 'features': {ft: ft_ix + i for i, ft in enumerate(unique)}}
            ft_ix += len(unique)
            continue

        raise ValueError(f'Column type [{col_type}] is not valid. Must be one of [label, con, cat]')

    return mapping_dict


def _write_ffm_file(output_file, arr, column_definitions, feature_mapping):
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
    with open(output_file, 'w') as f:
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
