# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import json


def load(service_url, service_type, db_connector):
    """
    DB에서 분석모델 메타 정보 가져오기

    Parameters
    ----------
    service_url: str
    service_type: str
    db_connector: DBConnector

    Returns
    -------
    dict
        메타 정보 딕셔너리
    """
    with db_connector as connected:
        query = connected.get_query('AiService_Module_HyperParameter', 'read')
        with connected.conn.cursor() as cursor:
            cursor.execute(query, (service_url, service_type,))
            result = cursor.fetchall()

    return json.loads(result[0][0])
