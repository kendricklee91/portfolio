# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>
import sentry_sdk
from db.query import get_query
from db.connector import MssqlConnector
import messages


def update_item_parameter(df):
    """
    문항 난이도, 변별도 DB 저장

    Parameters
    ----------
    df: pandas.DataFrame
        문항 난이도, 변별도 결과 데이터프레임

    Returns
    -------
    None
    """
    param_list = []
    for _, row in df.iterrows():
        param_list.append((
            row['item_id'],
            row['item_difficulty'],
            row['item_difficulty_code'],
            row['item_discrimination'],
            row['item_discrimination_code'],
        ))

    num_of_items = len(param_list)
    sentry_sdk.capture_message(
        messages.MSG_NUM_OF_ITEM.format(str(num_of_items)))

    if num_of_items > 0:
        _set_list('item', param_list)


def update_student_parameter(df):
    """
    학생 능력 DB 저장

    Parameters
    ----------
    df: pandas.DataFrame
        학생 능력 결과 데이터프레임

    Returns
    -------
    None
    """
    param_list = []
    for _, row in df.iterrows():
        param_list.append((
            row['student_id'],
            row['student_ability'],
        ))

    num_of_students = len(param_list)
    sentry_sdk.capture_message(
        messages.MSG_NUM_OF_STUDENT.format(str(num_of_students)))

    if num_of_students > 0:
        _set_list('student', param_list)


def _set_list(query_name, data_list):
    """
    데이터 리스트를 DB에 저장

    Parameters
    ----------
    query_name: str
        학생 능력 결과 데이터프레임
    data_list: list
        저장할 데이터 튜플 리스트

    Returns
    -------
    None
    """
    query = get_query(query_name, 'write')

    with MssqlConnector().connect() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, data_list)
        conn.commit()
