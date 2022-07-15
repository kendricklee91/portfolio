# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>


def postprocess(predicted_futures):
    """
    예측 결과 데이터를 후처리하여 Response JSON 규칙에 맞는 딕셔너리 리스트로 반환

    Parameters
    ----------
    predicted_futures: list_iterator
        List iterator of Futures
        (특정 Course, Skill에 대한 데이터셋 및 Model parameter, 예측 결과값 정보를 포함한 딕셔너리의 리스트를 반환하는 futures)

    Returns
    -------
    list:
        Response JSON 규칙에 맞는 결과값 데이터 딕셔너리의 리스트
    """
    postprocessed_list = []
    for future in predicted_futures:
        data = future.result()
        postprocessed_list.append(_postprocess(data['skill'], data['skill_idx'], data['infer_df'], data['profc_df']))

    return postprocessed_list


def _postprocess(skill, skill_idx, infer_df, profc_df):
    """
    특정 스킬의 예측 결과 데이터를 후처리하여 Response JSON 규칙에 맞는 딕셔너리로 만들어 반환

    Parameters
    ----------
    skill: str
        스킬
    skill_idx: int
        스킬 순서. 최상위 코드 여부를 체크하는 용도임.
    infer_df: pandas.DataFrame
        각 Skill 별 풀이이력 데이터 프레임
    profc_df: pandas.DataFrame
        각 Skill 별 예측 결과값 데이터 프레임

    Returns
    -------
    dict:
        Response JSON 규칙에 맞는 결과값 데이터 딕셔너리
    """
    infer_df = infer_df[:-1]

    if skill in ['cognitive', 'content']:
        infer_df['skill_cd'] = infer_df['skill_cd'].astype(int)

    def select_predicted(skill, tail):
        # 풀이이력 데이터 스킬 코드에 해당하는 예측 결과값을 추출
        code = tail['skill_cd'].values.item()
        idx = tail.index.item()
        result = {
            'code': code,
            'predicted': round(float(profc_df.loc[idx, skill]), 5)
        }
        return result

    predicted_list = infer_df.groupby('skill').apply(lambda x: select_predicted(x.name, x.tail(1))).tolist()

    if skill_idx == 0:  # 첫번째 스킬(단원 또는 성취도평가)인 경우 predicted value만 추가
        return predicted_list[0]

    return {'{}List'.format(skill): predicted_list}
