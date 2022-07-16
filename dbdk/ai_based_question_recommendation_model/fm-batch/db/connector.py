# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import pyodbc
import mysql.connector
from db import query_mssql
from db import query_mariadb


class DBConnector:
    """
    DB 커넥터

    Attributes
    ----------
    dbms: str
        DBMS 구분
    host: str
        DB 서버 호스트
    port: int
        DB 서버 포트
    db: str
        데이터베이스
    user: str
        DB 사용자
    passwd: str
        DB 비밀번호

    Methods
    -------
    connect()
        DB 연결. Connection 리턴
    _mssql_connect()
        MS SQL Server 연결
    _mysql_connect()
        MySQL 연결
    get_query()
    """

    def __init__(self, dbms, host, port, db, user, passwd):
        self.dialect = dbms.lower()
        self.host = host
        self.port = port
        self.db = db
        self.user = user
        self.passwd = passwd
        self.conn = None

        if self.dialect == 'mssql':
            self.queries = query_mssql.queries
            self.connect = self._mssql_connect
        elif self.dialect == 'mariadb':
            self.queries = query_mariadb.queries
            self.connect = self._mariadb_connect
        else:
            raise RuntimeError('DBMS {} is not supported'.format(self.dialect))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def get_query(self, query_name, crud, preparam=None):
        _query = self.queries[crud][query_name]

        if preparam:
            return _query.format(**preparam)

        return _query

    def _mssql_connect(self):
        self.conn = pyodbc.connect(
            'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={},{};DATABASE={};UID={};PWD={}'.format(self.host,
                                                                                                     self.port,
                                                                                                     self.db,
                                                                                                     self.user,
                                                                                                     self.passwd))

    def _mariadb_connect(self):
        self.conn = mysql.connector.connect(host=self.host, port=self.port, user=self.user, password=self.passwd,
                                            database=self.db)

    def __str__(self):
        return "dialect {} / host {} / port {} / db {} / user {} / passwd {}".format(
            self.dialect,
            self.host,
            self.port,
            self.db,
            self.user,
            self.passwd
        )
