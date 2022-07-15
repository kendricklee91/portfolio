# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import tensorflow as tf
import sentry_sdk
from model.dkt import dataloader
from model.dkt import preprocessor
from model.dkt import model
from model.dkt import postprocessor
import messages


def batch(course, skill, months, gpu):
    """
    DTK Batch Controller

    Parameters
    ----------
    course: str
        트레이닝 할 코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        트레이닝 할 스킬
    months: int
        트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치가 실행되는 주 시작일(일요일)로부터 계산.
    gpu: str
        사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.

    Returns
    -------
    None
    """
    num_of_gpus = _tf_init(gpu)  # Initialize Tensorflow

    # Loading
    sentry_sdk.add_breadcrumb(category='Loading')
    sentry_sdk.capture_message(messages.MSG_START.format('Loading'))
    df, skill_map = dataloader.load_data(course, skill, months)
    sentry_sdk.capture_message(messages.MSG_DF_SHAPE.format(df.shape))

    # Preprocessing
    sentry_sdk.add_breadcrumb(category='Preprocessing')
    sentry_sdk.capture_message(messages.MSG_START.format('Preprocessing'))
    train_set, valid_set, test_set = preprocessor.preprocess(course, skill, df, skill_map)

    # Training
    sentry_sdk.add_breadcrumb(category='Training')
    sentry_sdk.capture_message(messages.MSG_START.format('Training'))
    dkt = model.DKT(course, skill, train_set, valid_set, test_set, skill_map, num_of_gpus)
    dkt.train()
    sentry_sdk.capture_message(messages.MSG_MODEL_AUC.format(round(dkt.result_eval[2], 4)))

    # Postprocessing
    sentry_sdk.add_breadcrumb(category='Postprocessing')
    sentry_sdk.capture_message(messages.MSG_START.format('Postprocessing'))
    postprocessor.save_model(course, skill, dkt.result_model, skill_map)


def _tf_init(gpu):
    """
    GPU 사용을 위한 Tensorflow 초기화 작업 진행

    Parameters
    ----------
    gpu: str
        사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.

    Returns
    -------
    int
        선택된 GPU의 수
    """
    if gpu == "-1":
        physical_cpu_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_cpu_devices[0], 'CPU')
        return 0

    physical_gpu_devices = tf.config.list_physical_devices('GPU')
    if not physical_gpu_devices:
        raise RuntimeError(messages.MSG_GPU_NOT_SET.format(gpu))

    selected_gpu_devices = []
    gpu_indicies = [int(g) for g in gpu.split(',')]
    for gi in gpu_indicies:
        try:
            selected_gpu_devices.append(physical_gpu_devices[gi])
        except IndexError:
            raise RuntimeError(messages.MSG_GPU_NOT_SET.format(gpu))

    tf.config.set_visible_devices(selected_gpu_devices, 'GPU')

    return len(selected_gpu_devices)
