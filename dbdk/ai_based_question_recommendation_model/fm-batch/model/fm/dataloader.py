# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import pandas as pd


def load_data(query_name, columns, db_connector, subject):
    """
    FM 트레이닝을 위한 데이터 가져오기

    Parameters
    -------------------
    query_name: str
    columns: list
    db_connector: DBConnector
    subject: str

    Returns
    -------
    pandas.Dataframe
        문항 데이터
    """
    df = _read_df(db_connector, query_name, subject, columns)

    return df


def _read_df(db_connector, query_name, subject, columns):
    """
    DB 문항 데이터를 데이터 프레임으로 가져오기

    Parameters
    ----------
    db_connector: DBConnector
    query_name: str
    subject: str
    columns: list

    Returns
    -------
    pandas.Dataframe
        문항 데이터
    """
    with db_connector as connected:
        query = connected.get_query(query_name, 'read')
        df = pd.read_sql(query, connected.conn, params=[subject], columns=columns)

    return df
