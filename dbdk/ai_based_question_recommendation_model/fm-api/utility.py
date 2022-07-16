# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import pickle
import messages


def validate(validation, values):
    """
    요청 데이터 검증

    Parameters
    ----------
    validation: list
        검증 규칙
    values: dict
        요청 데이터

    Returns
    -------
    dict
        요청 데이터
    list
        검증 실패 시 오류메시지 목록, 성공 시 빈 리스트 반환.
    """
    params = {}
    errs = []
    if not values:
        errs.append({'param': '', 'msg': messages.MSG_EMPTY})
        return params, errs

    for v in validation:
        if v['param'] not in values:
            if v['required']:
                errs.append({'param': v['param'], 'msg': messages.MSG_MISSING})
                continue
            else:
                params[v['param']] = None
                continue

        params[v['param']] = values[v['param']]

        if 'type' in v:
            if type(values[v['param']]).__name__ != v['type']:
                errs.append({'param': v['param'], 'msg': ' '.join([messages.MSG_MUSTBE, v['type']])})
                continue

    return params, errs


def check_all_green(results):
    """
    결과값 리스트가 모두 성공인지 체크

    Parameters
    ----------
    results: list
        결과값 리스트

    Returns
    -------
    bool
        모두 성공인 경우 True를 반환
    """
    if not results:
        return False

    for result in results:
        if not result:
            return False

    return True


def deserialize_file(path):
    """
    pickled file 읽어오기

    Parameters
    ----------
    path: str

    Returns
    -------
    Any
        Deserialized data
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
