# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import sentry_sdk
from model.irt import dataloader
from model.irt import preprocessor
from model.irt.model import IRT
from model.irt import postprocessor
from model.irt import dataupdater
import messages


def batch(months, gpu):
    """
    IRT Batch Controller

    Parameters
    ----------
    months: int
        모델 트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치 실행일로부터 계산.
    gpu: str
        사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.

    Returns
    -------
    None
    """
    _tf_init(gpu)  # Initialize Tensorflow

    # Loading
    sentry_sdk.add_breadcrumb(category='Loading')
    sentry_sdk.capture_message(messages.MSG_START.format('Loading'))
    # 문항 풀이 이력 데이터 - 진단평가, 과목별 학교 공부, 성취도평가, 보충/심화, 쌍둥이 문제
    query_name_list = ['jindan', 'subject', 'achive', 'bogang', 'twin']
    df = dataloader.load_data(query_name_list, months)
    sentry_sdk.capture_message(messages.MSG_DF_SHAPE.format(df.shape))

    # Preprocessing
    sentry_sdk.add_breadcrumb(category='Preprocessing')
    sentry_sdk.capture_message(messages.MSG_START.format('Preprocessing'))
    preprocessed_df, seq_to_item, seq_to_student = preprocessor.preprocess(df)
    sentry_sdk.capture_message(messages.MSG_DF_SHAPE.format(preprocessed_df.shape))

    # Training
    sentry_sdk.add_breadcrumb(category='Training')
    sentry_sdk.capture_message(messages.MSG_START.format('Training'))
    item_df, student_df = IRT().bayesian_inference(preprocessed_df)
    sentry_sdk.capture_message(messages.MSG_DF_SHAPE.format(student_df.shape))
    sentry_sdk.capture_message(messages.MSG_DF_SHAPE.format(item_df.shape))

    # Postprocessing
    sentry_sdk.add_breadcrumb(category='Postprocessing')
    sentry_sdk.capture_message(messages.MSG_START.format('Postprocessing'))
    item_df, student_df = postprocessor.postprocess(item_df, student_df, seq_to_item, seq_to_student)

    # Update
    sentry_sdk.add_breadcrumb(category='Update')
    sentry_sdk.capture_message(messages.MSG_START.format('Update'))
    dataupdater.update_item_parameter(item_df)
    dataupdater.update_student_parameter(student_df)


def _tf_init(gpu):
    """
    GPU 사용을 위한 Tensorflow 초기화 작업 진행

    Parameters
    ----------
    gpu: str
        사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.

    Returns
    -------
    None
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if gpu != "-1":
        if len(device_lib.list_local_devices()) < 2:
            raise RuntimeError(messages.MSG_GPU_NOT_SET.format(gpu))

        # Set gpu memory limitation for multiple user
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)
