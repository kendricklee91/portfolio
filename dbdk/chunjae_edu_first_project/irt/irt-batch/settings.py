# 2020-05-12 Bokyong Seo<kay.seo@storypot.io>

# Database
MSSQL_HOST = "192.168.150.50"
MSSQL_PORT = 21433
MSSQL_DB = "HBEdu_App"
MSSQL_USER = "dbdk"
MSSQL_PW = "Vf!(98vP$@A_0LBR0I0aA"

# 센트리 설정
# Sentry 모니터링 서버의 접속 DSN. 형식: http://{Key}@{host}:{port}/{ProjectID}
SENTRY_DSN = "http://c7a08922ed1c40b292c807dfbf1df77c@183.110.210.105:9000/4"

# IRT 상수
IRT_DASK_NP_MULTIPLIER = 1
IRT_MIN_SCALE = 1e-1
IRT_STDDEV = 0.1
IRT_ALPHA_MIN = 0.0
IRT_ALPHA_MAX = 1.0
IRT_BETA_MIN = 0.0
IRT_BETA_MAX = 1.0
IRT_THETA_MIN = 0.0
IRT_THETA_MAX = 1.0
IRT_ITERATION = 2500
IRT_CRITICISM = False

# 문항 난이도 범위 : 하 < -0.5, -0.5 < 중 < 0.5, 상 > 0.5
# Boundary values
IRT_BND_DIFF_HIGH = 0.5
IRT_BND_DIFF_LOW = -0.5

# Boundary level values
IRT_LVL_DIFF_HIGH = 3
IRT_LVL_DIFF_MEDIUM = 2
IRT_LVL_DIFF_LOW = 1

# 문항 변별도 범위
# 변별도 거의 없는 문항 < 0.34
# 0.35 < 변별도 낮은 문항 < 0.64
# 0.65 < 변별도 적절한 문항 < 1.34
# 1.35 < 변별도 높은 문항 < 1.69
# 변별도 높은 문항 > 1.70
# Boundary values
IRT_BND_DISC_MAX = 1.70
IRT_BND_DISC_HIGH_MAX = 1.69
IRT_BND_DISC_HIGH_MIN = 1.35
IRT_BND_DISC_MEDIUM_MAX = 1.34
IRT_BND_DISC_MEDIUM_MIN = 0.65
IRT_BND_DISC_LOW_MAX = 0.64
IRT_BND_DISC_LOW_MIN = 0.35
IRT_BND_DISC_MIN = 0.34

# Boundary level values
IRT_LVL_DISC_MAX = 5
IRT_LVL_DISC_HIGH = 4
IRT_LVL_DISC_MEDIUM = 3
IRT_LVL_DISC_LOW = 2
IRT_LVL_DISC_MIN = 1
