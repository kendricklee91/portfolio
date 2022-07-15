# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import pymssql
import settings


class MssqlConnector():
    """
    DB 커넥터

    Attributes
    ----------
    hostname: str
        DB 서버 호스트
    port: int
        DB 서버 포트
    db: str
        데이터베이스
    user: str
        DB 사용자
    pw: str
        DB 비밀번호

    Methods
    -------
    connect()
        DB 연결. Connection을 리턴
    """
    def __init__(self):
        self.hostname = settings.MSSQL_HOST
        self.port = settings.MSSQL_PORT
        self.db = settings.MSSQL_DB
        self.user = settings.MSSQL_USER
        self.pw = settings.MSSQL_PW

    def connect(self):
        conn = pymssql.connect(host="%s:%s" % (
            self.hostname, self.port), user=self.user, password=self.pw, database=self.db)
        return conn
