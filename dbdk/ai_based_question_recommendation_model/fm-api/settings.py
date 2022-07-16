# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import json

# 환경변수로 부터 가져온 기초 설정값. docker-compose.yml 참조.
ENVS = {
    'AI_FM_URL': os.getenv('AI_FM_URL'),  # API 서비스 URL Path
    'AI_FM_SANIC_MODE': os.getenv('AI_FM_SANIC_MODE'),  # DEBUG or else
    'AI_FM_TYPE': 'A',  # 'A' - API, 'B' - Batch
    'AI_FM_DB': {
        'dbms': os.getenv('AI_FM_DB_DBMS'),  # 'mssql' or 'mariadb'
        'host': os.getenv('AI_FM_DB_HOST'),
        'port': int(os.getenv('AI_FM_DB_PORT')),
        'db': os.getenv('AI_FM_DB_DB'),
        'user': os.getenv('AI_FM_DB_USER'),
        'passwd': os.getenv('AI_FM_DB_PASSWD')
    }  # 데이터베이스 정보
}


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
