# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import os
import pandas as pd
import dask.dataframe as dd
from db.query import get_query
from db.connector import MssqlConnector
import settings
import messages


def load_data(course, skill, months):
    """
    배치에 사용할 풀이이력 데이터를 통계DB에서 가져옴
    코스에 따라 로직이 다름
    매일학교공부(thd): 매일학교공부 단일 쿼리를 가져오되 'user_id', 'skill_cd', 'correct' 컬럼만 가져옴
    성취도평가(ath): 성취도평가, 매일학교공부 각각에서 데이터를 가져와 _concat_ath_dfs 함수의 로직에 따라 병합

    Parameters
    ----------
    course: str
        트레이닝 할 코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        트레이닝 할 스킬
    months: int
        트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치가 실행되는 주 시작일(일요일)로부터 계산.

    Returns
    -------
    pandas.DataFrame
        배치에 사용할 풀이이력 데이터
    dict
        스킬코드 매핑 정보 딕셔너리
    """

    if course == 'thd':
        result_df = _read_df('thd', skill, months, ['user_id', 'skill_cd', 'correct'])

    if course == 'ath':
        thd_months = months - 6
        ath_df = _read_df('ath', skill, months)
        thd_df = _read_df('thd', skill, thd_months)
        result_df = _concat_ath_dfs(ath_df, thd_df, skill)

    skill_map = _read_map(course, skill)

    return result_df, skill_map


def _read_df(course, skill, months, columns=None):
    """
    DB에서 풀이이력 데이터를 데이터프레임으로 가져와, Dask 데이터프레임으로 변환 뒤, 사용자 UUID 타입을 정규화

    Parameters
    ----------
    course: str
        트레이닝 할 코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        트레이닝 할 스킬
    months: int
        데이터 기간
    columns: list
        컬럼 리스트. 없으면 쿼리에서 반환하는 모든 컬럼을 가져옴.

    Returns
    -------
    pandas.DataFrame
        배치에 사용할 풀이이력 데이터
    """
    query = get_query(course, 'read').format(settings.SKILL_COL_MAP[course][skill])

    with MssqlConnector().connect() as conn:
        df = pd.read_sql(query, conn, params={'months': months}, columns=columns)

    if len(df.index) == 0:
        raise RuntimeError(messages.MSG_DF_EMPTY)

    df['user_id'] = df['user_id'].astype(str)  # 사용자 UUID 타입을 string으로 정규화

    return df


def _read_map(course, skill):
    """
    DB에서 스킬코드 매핑 정보를 반환

    Parameters
    ----------
    course: str
        코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        스킬

    Returns
    -------
    dict
        매핑 정보 딕셔너리
    """
    query = get_query('map', 'read')

    with MssqlConnector().connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, {'course': course, 'skill': skill})
            if cursor.rowcount == 0:
                raise RuntimeError(messages.MSG_DF_EMPTY)

            records = cursor.fetchall()

    map_skill_factorval = {}
    map_skill_standard = {}
    for row in records:
        map_skill_factorval[row[0]] = row[2]
        map_skill_standard[row[0]] = row[1]

    max_factorval = _read_max_factorval(course, skill)

    return {
        'map_skill_factorval': map_skill_factorval,
        'map_skill_standard': map_skill_standard,
        'skill_depth': max_factorval + 1,
        'features_depth': (max_factorval + 1) * 2
    }


def _read_max_factorval(course, skill):
    """
    DB에서 Skill factorized value 최대값 반환

    Parameters
    ----------
    course: str
        코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        스킬

    Returns
    -------
    int
        Skill factorized value 최대값
    """
    query = get_query('map_max_factorval', 'read')

    with MssqlConnector().connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, {'course': course, 'skill': skill})
            if cursor.rowcount == 0:
                raise RuntimeError(messages.MSG_DF_EMPTY)

            record = cursor.fetchone()
            return record[0]


def _concat_ath_dfs(ath_df, thd_df, skill):
    """
    성취도평가를 풀이한 학생 및 문항의 단원 정보로 매일학교공부 데이터를 필터링하여, 학생별로 성취도평가 데이터와 병합
    Dask를 사용한 병렬처리

    Parameters
    ----------
    ath_df: pandas.DataFrame
        성취도평가 데이터프레임
    thd_df: pandas.DataFrame
        매일학교공부 데이터프레임
    skill: str
        스킬

    Returns
    -------
    pandas.DataFrame
        배치에 사용할 풀이이력 데이터프레임
    """
    thd_ddf = dd.from_pandas(thd_df, npartitions=int(os.cpu_count() / 2))  # Use dask

    ath_uid_list = ath_df['user_id'].unique().tolist()
    ath_lessoncd_list = ath_df['lesson_cd'].unique().tolist()

    thd_ddf = thd_ddf[thd_ddf['user_id'].isin(ath_uid_list)]
    thd_ddf = thd_ddf[thd_ddf['lesson_cd'].isin(ath_lessoncd_list)]

    def _uid_filter(userid, thd_uid_df):
        # 각 학생별로, 문항의 단원 정보로 매일학교공부 데이터를 필터링하여, 학생별로 성취도평가 데이터와 병합
        ath_uid_df = ath_df[ath_df['user_id'] == userid]

        result_df_list = []
        for achieve_cd in ath_uid_df['achieve_cd'].unique().tolist():
            ath_uid_achieve_df = ath_uid_df[ath_uid_df['achieve_cd'] == achieve_cd].copy()
            ath_uid_lessoncd_list = ath_uid_achieve_df['lesson_cd'].unique().tolist()
            thd_subset_df = thd_uid_df[thd_uid_df['lesson_cd'].isin(ath_uid_lessoncd_list)].copy()

            if skill == 'achieve':
                thd_subset_df['skill_cd'] = achieve_cd

            result_df = pd.concat([thd_subset_df, ath_uid_achieve_df])
            result_df_list.append(result_df)

        return pd.concat(result_df_list)

    result_ddf = thd_ddf.groupby('user_id').apply(lambda x: _uid_filter(x.name, x), meta=thd_ddf)
    result_df = result_ddf.compute()

    return result_df.reset_index(drop=True)
