# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import tensorflow as tf
import pandas as pd
import numpy as np
import sentry_sdk
import settings
import messages


def preprocess(course, skill, df, skill_map):
    """
    DKT 모델에 입력하기 위해 데이터 전처리 및 모델 실행을 위한 변수 생성

    Parameters
    ----------
    course: str
        트레이닝 할 코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        트레이닝 할 스킬
    df: pandas.DataFrame
        배치에 사용할 풀이이력 데이터프레임
    skill_map: dict
        스킬코드 매핑 정보 딕셔너리

    Returns
    -------
    tf.data.Dataset
        Train dataset
    tf.data.Dataset
        Validation dataset
    tf.data.Dataset
        Test dataset
    """
    if 'user_id' not in df.columns:
        raise KeyError(messages.MSG_COLUMN_REQUIED.format('user_id'))
    if 'skill_cd' not in df.columns:
        raise KeyError(messages.MSG_COLUMN_REQUIED.format('skill_cd'))
    if 'correct' not in df.columns:
        raise KeyError(messages.MSG_COLUMN_REQUIED.format('correct'))

    df.dropna(subset=['skill_cd'], inplace=True)
    df = df.groupby('user_id').filter(lambda q: len(q) > 1)
    sentry_sdk.capture_message(messages.MSG_TOTAL_USER.format(len(df['user_id'].unique())))

    df['correct'] = df['correct'].replace({'O': 1}).replace({'X': 0})
    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(messages.MSG_COLUMN_CORRECT)

    df['skill_cd'] = df['skill_cd'].astype(str)  # Skill code - Factorized value 매핑을 위한 string 캐스팅
    df['skill'] = df['skill_cd'].replace(skill_map['map_skill_factorval'])
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    skill_depth = skill_map['skill_depth']
    features_depth = skill_map['features_depth']

    train_df, test_df = _split_train_test(df, settings.PREP_TEST_FRACTION, settings.PREP_RANDOM_STATE_VALUE)

    train_dataset, train_users = _get_dataset(
        train_df, features_depth, skill_depth, settings.PREP_BATCH_SIZE, settings.PREP_SHUFFLE)
    test_dataset, test_users = _get_dataset(test_df, features_depth, skill_depth,
                                            settings.PREP_BATCH_SIZE, settings.PREP_SHUFFLE)

    length = train_users // settings.PREP_BATCH_SIZE
    length_rest = train_users % settings.PREP_BATCH_SIZE
    train_length = length + length_rest

    # train + valid dataset size based on number of train_users.
    total_set_size = length * settings.PREP_BATCH_SIZE + length_rest
    # size of validation dataset 20% of total dataset size.
    valid_set_size = round(total_set_size * settings.PREP_VAL_FRACTION)
    train_set_size = round(total_set_size - valid_set_size)

    sentry_sdk.capture_message(messages.MSG_SET_SIZES.format(train_set_size, valid_set_size, test_users))
    sentry_sdk.capture_message(messages.MSG_NUM_OF_FEATURES.format(features_depth, skill_depth))

    train_set, valid_set = _split_train_val(train_dataset, train_length, settings.PREP_VAL_FRACTION)
    return train_set, valid_set, test_dataset


def _split_train_test(df, test_fraction, random_state_value):
    """
    입력된 skill의 데이터프레임을 train과 test로 분할 (유니크 학생을 비율로 분할)

    Parameters
    ----------
    df: pandas.DataFrame
        전처리된 풀이이력 데이터프레임
    test_fraction: float
        Test fraction
    random_state_value: int
        Random state

    Returns
    -------
    pandas.DataFrame
        Train dataframe
    pandas.DataFrame
        Test dataframe
    """
    total_uid_df = pd.DataFrame(df['user_id'].unique())
    test_uid_df = total_uid_df.sample(frac=test_fraction, random_state=random_state_value)
    test_uid_values = test_uid_df[0].values

    train_df = df[~df['user_id'].isin(test_uid_values)].reset_index(drop=True)
    test_df = df[df['user_id'].isin(test_uid_values)].reset_index(drop=True)

    return train_df, test_df


def _get_dataset(df, features_depth, skill_depth, batch_size, shuffle=None):
    """
    skill, correct, skill_with_answer에 대해 각각의 시퀀스 범위 조정
    모델에 입력하기 위해 tf.data.Dataset 타입에 맞게 변환 작업
    배치 사이즈 마다 패딩 작업 진행

    Parameters
    ----------
    df: pandas.DataFrame
        전처리된 풀이이력 데이터프레임
    features_depth: float
        Features depth
    skill_depth: int
        Skill depth
    batch_size: int
        Batch size
    shuffle: bool
        Should shuffle?

    Returns
    -------
    tf.data.Dataset
        데이터셋
    int
        데이터셋의 유저수
    """
    seq = df.groupby('user_id').apply(lambda r: (
        r['skill_with_answer'].values[:-1], r['skill'].values[1:], r['correct'].values[1:],))
    users = len(seq)

    dataset = tf.data.Dataset.from_generator(generator=lambda: seq, output_types=(tf.int32, tf.int32, tf.float32))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=users)

    dataset = dataset.map(lambda feat, skill, label: (tf.one_hot(feat, depth=features_depth), tf.concat(
        values=[tf.one_hot(skill, depth=skill_depth), tf.expand_dims(label, -1)], axis=-1)))
    dataset = dataset.padded_batch(batch_size=batch_size, padding_values=(
        settings.PREP_MASK_VALUE, settings.PREP_MASK_VALUE), padded_shapes=([None, None], [None, None]), drop_remainder=True)
    return dataset, users


def _split_train_val(dataset, total_size, val_fraction):
    """
    train 데이터셋을 train / validation 데이터셋으로 다시 분할

    Parameters
    ----------
    dataset: pandas.DataFrame
        전처리된 풀이이력 데이터프레임
    total_size: int
        Total length
    val_fraction: float
        Validation fraction

    Returns
    -------
    tf.data.Dataset
        Train dataset
    tf.data.Dataset
        Validation dataset
    """
    val_size = np.ceil(val_fraction * total_size)
    train_set, valid_set = _split(dataset, val_size)
    return train_set, valid_set


def _split(dataset, split_size):
    splitset = dataset.take(split_size)
    dataset = dataset.skip(split_size)
    return dataset, splitset
