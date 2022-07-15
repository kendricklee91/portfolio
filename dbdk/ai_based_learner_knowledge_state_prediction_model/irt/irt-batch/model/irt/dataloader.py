# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>
import pandas as pd
from db.query import get_query
from db.connector import MssqlConnector
import messages

# 문항 풀이 이력 읽기
def load_data(query_name_list, months):
    """
    배치에 사용할 풀이이력 데이터를 통계DB에서 가져옴

    Parameters
    ----------
    query_name_list: list
        데이터를 가져올 query.py의 쿼리명 목록
    months: int
        모델 트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치 실행일로부터 계산.

    Returns
    -------
    pandas.DataFrame
        배치에 사용할 풀이이력 데이터
    """
    raw_data = []
    with MssqlConnector().connect() as conn:
        for name in query_name_list:
            query = get_query(name, 'read')
            df = pd.read_sql(query, conn, params={'months': months})
            raw_data.append(df)

    result_df = pd.concat(raw_data, ignore_index=True)

    if len(result_df.index) == 0:
        raise RuntimeError(messages.MSG_DF_EMPTY)

    return result_df
