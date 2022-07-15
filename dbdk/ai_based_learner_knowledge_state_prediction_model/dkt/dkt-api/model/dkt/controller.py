# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import sentry_sdk
from model.dkt import dataloader
from model.dkt import preprocessor
from model.dkt import model
from model.dkt import postprocessor
import messages


def inference(course, skills, data, model_params):
    """
    DKT Inference Controller

    Parameters
    ----------
    course: str
        코스. 단원마무리-'thd' / 성취도평가-'ath'
    skills: list
        스킬 리스트
    data: dict
        요청 데이터(풀이이력)
    modeL_params: dict
        각 Course, Skill 별로 Model parameter를 담고 있는 딕셔너리

    Returns
    -------
    dict
        응답 데이터
    """
    result = {
        'userid': data['userid'],
        skills[0]: data[skills[0]]
    }

    try:
        df = dataloader.load_data(skills, data)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        result['result'] = False
        result['errs'] = [{'msg': messages.MSG_DATA_LOADING_ERR}]
        return result

    try:
        preprocessed_list = preprocessor.preprocess(course, skills, model_params, df)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        result['result'] = False
        result['errs'] = [{'msg': messages.MSG_DATA_PREPROCESSING_ERR}]
        return result

    try:
        predicted_futures = model.predict(preprocessed_list)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        result['result'] = False
        result['errs'] = [{'msg': messages.MSG_DATA_MODEL_ERR}]
        return result

    try:
        postprocessed_list = postprocessor.postprocess(predicted_futures)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        result['result'] = False
        result['errs'] = [{'msg': messages.MSG_DATA_POSTPROCESSING_ERR}]
        return result

    result['result'] = True
    for postprocessed in postprocessed_list:
        result.update(postprocessed)

    return result
