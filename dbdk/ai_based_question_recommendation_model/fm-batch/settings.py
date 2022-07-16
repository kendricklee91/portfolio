# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import json
from db.connector import DBConnector

# Database
# DBMS supports: mssql, mysql
SETTING_DB_SETTINGS = {
    'dbms': 'mssql',
    'host': '192.168.150.30',
    'port': 1433,
    'db': 'HBEdu_App_Study',
    'user': 'AiSystem',
    'passwd': 'Jn!@Ea_$Aqll6J(JumkDq'
}

STATISTICS_DB_SETTINGS = {
    'dbms': 'mssql',
    'host': '192.168.150.50',
    'port': 21433,
    'db': 'HBEdu_App',
    'user': 'dbdk',
    'passwd': 'Vf!(98vP$@A_0LBR0I0aA'
}

# 센트리 설정
# Sentry 모니터링 서버의 접속 DSN. 형식: http://{Key}@{host}:{port}/{ProjectID}
SENTRY_DSN = "http://8b9ae1bd5487450fbe4f14de9d293168@183.110.210.106:9000/3"


def load(service_url, service_type, db_connector):
    """
    DB에서 설정정보 가져오기
    Parameters
    ----------
    service_url: str
    service_type: str
    db_connector: DBConnector

    Returns
    -------
    dict
        설정값 딕셔너리
    """
    with db_connector as connected:
        query = connected.get_query('AiService_Module_Settings', 'read')
        with connected.conn.cursor() as cursor:
            cursor.execute(query, (service_url, service_type,))
            result = cursor.fetchall()

    return json.loads(result[0][0])
