# Directory paths
DATA_PATH = '/home/ubuntu/workspace/taxfriend/taxf_data'
MODEL_PATH = '/home/ubuntu/workspace/taxfriend/taxf_model'
LOG_PATH = '/home/ubuntu/workspace/taxfriend/taxf_logs'

# Preprocessing parameters
USING_NM_ITEM = True
WORD_FREQ = 300
REPLACE = {
    '�' : '',
    '\u3000' : '',
    '，' : ',',
    '・' : ',',
    'ㆍ' : ',',
    '·' : ',',
    '．' : ',',
    '.' : ',',
    '\+' : ''
}

# Model parameters
TEST_SIZE = 0.3
THRESHOLD = 0.8
TUNING = False