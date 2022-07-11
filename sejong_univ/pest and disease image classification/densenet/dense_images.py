from dense_packages import *

# Grape
# Train (8), Valid (1), Test (1)
A_TRAIN_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_train' # 80%
A_VALID_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_valid' # 10%
A_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_test' # 10%

ANTH_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_anthracnose' # 탄저병 test 데이터
BACT_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_bacterialspot' # 세균점무늬병 test 데이터
CMV_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_cmv' # 오이모자이크바이러스 test 데이터
GRAY_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_graymold' # 잿빛곰팡이병 test 데이터
TSWV_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/crop_tswv' # 토마토반점위조바이러스 test 데이터
NORM_TEST_DIR = 'C:/Users/lee/Desktop/grad_paper7/disease_img_han/pepper/A/\crop_normal' # 정상 test 데이터

train_file_list = os.listdir(A_TRAIN_DIR)
valid_file_list = os.listdir(A_VALID_DIR)
test_file_list = os.listdir(A_TEST_DIR)

anth_file_list = os.listdir(ANTH_TEST_DIR)
bact_file_list = os.listdir(BACT_TEST_DIR)
cmv_file_list = os.listdir(CMV_TEST_DIR)
gray_file_list = os.listdir(GRAY_TEST_DIR)
tswv_file_list = os.listdir(TSWV_TEST_DIR)
norm_file_list = os.listdir(NORM_TEST_DIR)

IMG_SIZE = 32

def img_label(lbl):
    disease_label = lbl.split('_')[0] + "_" + lbl.split('_')[1]

    # 배
    if disease_label == 'pepper_anthracnose': # 고추 - 탄저병
        return [1, 0, 0, 0, 0, 0] # 0
    elif disease_label == 'pepper_bacterialspot': # 고추 - 세균점무늬병
        return [0, 1, 0, 0, 0, 0] # 1
    elif disease_label == 'pepper_cmv': # 포도 - 오이모자이크바이러스
        return [0, 0, 1, 0 ,0, 0] # 2
    elif disease_label == 'pepper_graymold': # 포도 - 잿빛곰팡이병
        return [0, 0, 0, 1, 0, 0] # 3
    elif disease_label == 'pepper_tswv': # 포도 - 토마토반점위조바이러스
        return [0, 0, 0, 0, 1, 0] # 4
    elif disease_label == 'pepper_normal': # 정상
        return [0, 0, 0, 0, 0, 1] # 5

#########################################################

# train용 이미지 데이터
def create_train_data():
    train_data = []

    for img in train_file_list:
        path = os.path.join(A_TRAIN_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        train_data.append(img)

    train_img_array = np.asarray(train_data)
    return train_img_array

# train용 이미지 레이블
def create_train_label():
    train_label = []

    for lbl in train_file_list:
        label = img_label(lbl)
        train_label.append(label)

    train_lbl_array = np.asarray(train_label)
    return train_lbl_array

#########################################################

# valid용 이미지 데이터
def create_valid_data():
    valid_data = []

    for img in valid_file_list:
        path = os.path.join(A_VALID_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        valid_data.append(img)

    valid_img_array = np.asarray(valid_data)
    return valid_img_array

# valid용 이미지 레이블
def create_valid_label():
    valid_label = []

    for lbl in valid_file_list:
        label = img_label(lbl)
        valid_label.append(label)

    valid_lbl_array = np.asarray(valid_label)
    return valid_lbl_array

#########################################################

# test용 이미지 데이터
def create_test_data():
    test_data = []

    for img in test_file_list:
        path = os.path.join(A_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_data.append(img)

    test_img_array = np.asarray(test_data)
    return test_img_array

# test용 이미지 레이블
def create_test_label():
    test_label = []

    for lbl in test_file_list:
        label = img_label(lbl)
        test_label.append(label)

    test_lbl_array = np.asarray(test_label)
    return test_lbl_array

#########################################################

# 탄저병 test용 이미지 데이터
def create_anth_data():
    anth_data = []

    for img in anth_file_list:
        path = os.path.join(ANTH_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        anth_data.append(img)

    anth_img_array = np.asarray(anth_data)
    return anth_img_array

# 탄저병 test용 이미지 레이블
def create_anth_label():
    anth_label = []

    for lbl in anth_file_list:
        label = img_label(lbl)
        anth_label.append(label)

    anth_lbl_array = np.asarray(anth_label)
    return anth_lbl_array

#########################################################

# 세균점무늬병 test용 이미지 데이터
def create_bact_data():
    bact_data = []

    for img in bact_file_list:
        path = os.path.join(BACT_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        bact_data.append(img)

    bact_img_array = np.asarray(bact_data)
    return bact_img_array

# 세균점무늬병 test용 이미지 레이블
def create_bact_label():
    bact_label = []

    for lbl in bact_file_list:
        label = img_label(lbl)
        bact_label.append(label)

    bact_lbl_array = np.asarray(bact_label)
    return bact_lbl_array

#########################################################

# cmv test용 이미지 데이터
def create_cmv_data():
    cmv_data = []

    for img in cmv_file_list:
        path = os.path.join(CMV_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cmv_data.append(img)

    cmv_img_array = np.asarray(cmv_data)
    return cmv_img_array

# cmv test용 이미지 레이블
def create_cmv_label():
    cmv_label = []

    for lbl in cmv_file_list:
        label = img_label(lbl)
        cmv_label.append(label)

    cmv_lbl_array = np.asarray(cmv_label)
    return cmv_lbl_array

#########################################################

# 잿빛곰팡이병 test용 이미지 데이터
def create_gray_data():
    gray_data = []

    for img in gray_file_list:
        path = os.path.join(GRAY_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray_data.append(img)

    gray_img_array = np.asarray(gray_data)
    return gray_img_array

# 잿빛곰팡이병 test용 이미지 레이블
def create_gray_label():
    gray_label = []

    for lbl in gray_file_list:
        label = img_label(lbl)
        gray_label.append(label)

    gray_lbl_array = np.asarray(gray_label)
    return gray_lbl_array

#########################################################

# 흰가루병 test용 이미지 데이터
def create_tswv_data():
    tswv_data = []

    for img in tswv_file_list:
        path = os.path.join(TSWV_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tswv_data.append(img)

    tswv_img_array = np.asarray(tswv_data)
    return tswv_img_array

# 흰가루병 test용 이미지 레이블
def create_tswv_label():
    tswv_label = []

    for lbl in tswv_file_list:
        label = img_label(lbl)
        tswv_label.append(label)

    tswv_lbl_array = np.asarray(tswv_label)
    return tswv_lbl_array

########################################################

# 정상 test용 이미지 데이터
def create_norm_data():
    norm_data = []

    for img in norm_file_list:
        path = os.path.join(NORM_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        norm_data.append(img)

    norm_img_array = np.asarray(norm_data)
    return norm_img_array

# 정상 test용 이미지 레이블
def create_norm_label():
    norm_label = []

    for lbl in norm_file_list:
        label = img_label(lbl)
        norm_label.append(label)

    norm_lbl_array = np.asarray(norm_label)
    return norm_lbl_array