# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import os

# API 서버 설정
API_HOST = "0.0.0.0"
API_PORT = 3000

# Sanic 설정
REQUEST_MAX_SIZE = 10 * 1024 * 1024  # How big a request may be (bytes)
REQUEST_TIMEOUT = 30  # How long a request can take to arrive (sec)
RESPONSE_TIMEOUT = 30  # How long a response can take to process (sec)
SANIC_WORKERS = int(os.cpu_count() * 0.7)

# 센트리 설정
# Sentry 모니터링 서버의 접속 DSN. 형식: http://{Key}@{host}:{port}/{ProjectID}
SENTRY_DSN = "http://7fb392d60e264b5e956a105823a9deca@183.110.210.106:9000/5"

# 모델 Git 리모트 레포지토리
GIT_MODEL_REPO = "ssh://git@183.110.210.105:2224/root/dkt-model.git"

# 각 Course에 포함되는 Skill 설정
SKILL_CODES = {
    'thd': ['lesson', 'lecture', 'cognitive', 'difficulty', 'ptype'],
    'ath': ['achieve', 'lesson', 'cognitive', 'content', 'difficulty', 'ptype']
}

PREP_MASK_VALUE = -1.  # The masking value cannot be zero.

# DKT Model Path
BASE_DIR = '/data/dkt'
PERSISTENT_DIR = os.path.join(BASE_DIR, 'persistent')

MODEL_SKILL_MAP_PATH = os.path.join(PERSISTENT_DIR, '{}_{}_skill_map.json')

# DKT Tensorflow Serving URI
MODEL_TS_URI = 'http://localhost:8501/v1/models/{}_{}/versions/1:predict'
