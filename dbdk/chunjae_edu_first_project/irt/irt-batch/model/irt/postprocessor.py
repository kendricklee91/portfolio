# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>
import settings


def postprocess(item_df, student_df, seq_to_item, seq_to_student):
    """
    추론된 결과를 출력 데이터로 변환

    Parameters
    ----------
    item_df: pandas.DataFrame
        문항 난이도, 변별도 결과 데이터프레임
    student_df: pandas.DataFrame
        학생 능력 결과 데이터프레임
    seq_to_item: dict
        시퀀스와 학생ID의 매핑 테이블
    seq_to_student: dict
        시퀀스와 문항ID의 매핑 테이블

    Returns
    -------
    pandas.DataFrame
        출력데이터로 변환된 문항 난이도, 변별도 결과 데이터프레임
    pandas.DataFrame
        출력데이터로 변환된 학생 능력 결과 데이터프레임
    """
    # 문항코드와 학생ID를 원본 데이터의 값으로 변환
    item_df['item_id'] = item_df['item_id'].map(seq_to_item)
    student_df['student_id'] = student_df['student_id'].map(seq_to_student)

    # 문항 난이도/변별도 등급 변경
    item_df['item_difficulty_code'] = item_df['item_difficulty'].apply(
        _item_diff_grade)
    item_df['item_discrimination_code'] = item_df['item_discrimination'].apply(
        _item_disc_grade)

    return item_df, student_df


def _item_diff_grade(x):
    """
    문항 난이도 범위 설정값(settings.py)에 따라 난이도 등급을 반환
    난이도 범위 : 1 ~ 3

    Parameters
    ----------
    x: float
        문항 난이도

    Returns
    -------
    int
        문항 난이도 등급
    """
    if x > settings.IRT_BND_DIFF_HIGH:
        return settings.IRT_LVL_DIFF_HIGH
    elif settings.IRT_BND_DIFF_LOW < x < settings.IRT_BND_DIFF_HIGH:
        return settings.IRT_LVL_DIFF_MEDIUM
    else:
        return settings.IRT_LVL_DIFF_LOW


def _item_disc_grade(x):
    """
    문항 변별도 범위 설정값(settings.py)에 따라 변별도 등급을 반환
    변별도 범위 : 1 ~ 5

    Parameters
    ----------
    x: float
        문항 변별도

    Returns
    -------
    int
        문항 변별도 등급
    """
    if x > settings.IRT_BND_DISC_MAX:
        return settings.IRT_LVL_DISC_MAX
    elif settings.IRT_BND_DISC_HIGH_MIN < x < settings.IRT_BND_DISC_HIGH_MAX:
        return settings.IRT_LVL_DISC_HIGH
    elif settings.IRT_BND_DISC_MEDIUM_MIN < x < settings.IRT_BND_DISC_MEDIUM_MAX:
        return settings.IRT_LVL_DISC_MEDIUM
    elif settings.IRT_BND_DISC_LOW_MIN < x < settings.IRT_BND_DISC_LOW_MAX:
        return settings.IRT_LVL_DISC_LOW
    else:
        return settings.IRT_LVL_DISC_MIN
