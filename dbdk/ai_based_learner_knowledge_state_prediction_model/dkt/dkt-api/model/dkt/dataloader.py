# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import pandas as pd


def load_data(skills, data):
    """
    요청 데이터(풀이이력)를 데이터 프레임으로 변환

    Parameters
    ----------
    skills: list
        스킬 리스트
    data: dict
        요청 데이터

    Returns
    -------
    pandas.DataFrame
        풀이이력 데이터 프레임
    """
    columns = ['user_id', 'correct']
    columns.extend(skills)

    values_list = _reorder(skills, data)
    df = pd.DataFrame(values_list, columns=columns)

    return df


def _reorder(skills, data):
    """
    요청 데이터(풀이이력)를 2차원 리스트로 재배열

    Parameters
    ----------
    skills: list
        스킬 리스트
    data: dict
        요청 데이터

    Returns
    -------
    list
        풀이이력 리스트
    """
    values_list = []
    user_id = data['userid']
    first_cd = data[skills[0]]  # 최상위 스킬코드: 단원마무리(thd) - lesson / 성취도평가(ath) - achieve

    for second in data['{}List'.format(skills[1])]:
        second_cd = second[skills[1]]   # 차상위 스킬코드: 단원마무리(thd) - lecture / 성취도평가(ath) - lesson
        for question in second['questionList']:
            values = [user_id, question['correct'], first_cd, second_cd]
            for skill in skills:
                if skill in question:
                    values.append(question[skill])

            values_list.append(values)

    return values_list
