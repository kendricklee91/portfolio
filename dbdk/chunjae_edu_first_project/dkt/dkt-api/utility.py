# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
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
    list
        검증 실패 시 오류메시지 목록, 성공 시 빈 리스트 반환.
    """
    errs = []
    if not values:
        errs.append({'param': '', 'msg': messages.MSG_EMPTY})
        return errs

    for v in validation:
        if v['param'] not in values:
            errs.append({'param': v['param'], 'msg': messages.MSG_MISSING})
            continue

        if 'type' in v:
            if not isinstance(values[v['param']], v['type']):
                errs.append({'param': v['param'], 'msg': ' '.join(
                    [messages.MSG_MUSTBE, v['type'].__name__])})
                continue

        if 'regex' in v:
            if not v['regex'].match(str(values[v['param']])):
                errs.append({'param': v['param'], 'msg': ' '.join(
                    [messages.MSG_MUSTBE, v['regex'].pattern])})
                continue

        if isinstance(values[v['param']], list) and 'child' in v:
            child_list = values[v['param']]
            if not child_list:
                errs.append({'param': v['param'], 'msg': messages.MSG_EMPTY})
                continue

            for item in child_list:
                errs.extend(validate(v['child'], item))

        if isinstance(values[v['param']], dict) and 'child' in v:
            child_dict = values[v['param']]
            if not child_dict:
                errs.append({'param': v['param'], 'msg': messages.MSG_EMPTY})
                continue

            errs.extend(validate(v['child'], child_dict))

    return errs


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
