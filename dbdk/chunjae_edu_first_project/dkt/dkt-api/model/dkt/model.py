# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import concurrent.futures
import pandas as pd
import requests
import json


def predict(preprocessed_list):
    """
    DKT 예측

    Parameters
    ----------
    preprocessed_list: list
        전처리 데이터 딕셔너리의 리스트
        (특정 Course, Skill에 대한 데이터셋 및 Model parameter 등 정보를 포함한 딕셔너리의 리스트)

    Returns
    -------
    list_iterator:
        List iterator of Futures
        (특정 Course, Skill에 대한 데이터셋 및 Model parameter, 예측 결과값 정보를 포함한 딕셔너리의 리스트를 반환하는 futures)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(preprocessed_list)) as executor:
        future_to_predict = {executor.submit(_predict, data): data for data in preprocessed_list}
        predicted_futures = concurrent.futures.as_completed(future_to_predict)

    return predicted_futures


def _predict(data):
    """
    특정 스킬에 대해 Tensorflow Serving에 예측 요청 및 예측 결과를 preprocessed_data 딕셔너리에 추가하여 반환

    Parameters
    ----------
    preprocessed_data: dict
        전처리 데이터 딕셔너리)
        (특정 Course, Skill에 대한 데이터셋 및 Model parameter 등 정보를 포함한 딕셔너리)

    Returns
    -------
    dict:
        preprocessed_data에 예측 결과("profc_df")를 추가
    """
    serving_uri = data['model_param']['serving_uri']
    request_json = json.dumps({"instances": data['request_data']})
    json_response = requests.post(serving_uri, data=request_json, headers={"content-type": "application/json"})

    result = json.loads(json_response.text)
    if 'error' in result:
        raise RuntimeError(result['error'])

    profc_df = pd.DataFrame(result['predictions'][0])

    predicted_data = data.copy()
    predicted_data.update({'profc_df': profc_df})

    return predicted_data
