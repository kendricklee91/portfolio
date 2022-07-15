# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import os

# Database
MSSQL_HOST = "192.168.150.50"
MSSQL_PORT = 21433
MSSQL_DB = "HBEdu_App"
MSSQL_USER = "dbdk"
MSSQL_PW = "Vf!(98vP$@A_0LBR0I0aA"

# 센트리 설정
# Sentry 모니터링 서버의 접속 DSN. 형식: http://{Key}@{host}:{port}/{ProjectID}
SENTRY_DSN = "http://422e8fb3ccfd4fda90855f6e136b584a@183.110.210.106:9000/6"

# 모델 Git 리모트 레포지토리
GIT_MODEL_REPO = "ssh://git@183.110.210.105:2224/root/dkt-model.git"

# DKT 상수
# Data Loading
SKILL_COL_MAP = {
    'thd': {
        'achieve': 'ulcms.LecCode',
        'lesson': 'ulcms.LecCode',
        'lecture': 'llcms.MCode',
        'cognitive': 'tpa.f_analysis_id',
        'content': 'tpa.f_studytree_id',
        'difficulty': 'tpa.f_difficult_cd',
        'ptype': 'tp.f_ptype_cd'
    },
    'ath': {
        'achieve': 'atdm.f_achieve_cd',
        'lesson': 'atdm.LecCode',
        'cognitive': 'atdm.f_analysis_id',
        'content': 'atdm.f_studytree_id',
        'difficulty': 'atdm.f_difficult_cd',
        'ptype': 'atdm.f_ptype_cd'
    }
}

# Preprocessing
PREP_RANDOM_STATE_VALUE = 10
PREP_TEST_FRACTION = 0.2
PREP_VAL_FRACTION = 0.2
PREP_BATCH_SIZE = 20
PREP_SHUFFLE = True
PREP_MASK_VALUE = -1.  # The masking value cannot be zero.

# Hyper Parameters
BATCH_OPTIONS = {
    "thd": {
        "lesson": {
            "learning_rate": 0.0007,
            "hidden_units": 100,
            "dropout_rate": 0.4,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "lecture": {
            "learning_rate": 0.0006,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 50
        },
        "cognitive": {
            "learning_rate": 0.0006,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 30
        },
        "difficulty": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.4,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 50
        },
        "ptype": {
            "learning_rate": 0.0007,
            "hidden_units": 100,
            "dropout_rate": 0.4,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        }
    },
    "ath": {
        "achieve": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "lesson": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "cognitive": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "content": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "difficulty": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        },
        "ptype": {
            "learning_rate": 0.0003,
            "hidden_units": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "callbacks": None,
            "shuffle": True,
            "verbose": 0,
            "epochs": 40
        }
    }
}

# DKT Model Path
BASE_DIR = '/data/dkt'
PERSISTENT_DIR = os.path.join(BASE_DIR, 'persistent')

MODEL_PATH = os.path.join(PERSISTENT_DIR, 'model/{}/{}/1')
MODEL_SKILL_MAP_PATH = os.path.join(PERSISTENT_DIR, '{}_{}_skill_map.json')
