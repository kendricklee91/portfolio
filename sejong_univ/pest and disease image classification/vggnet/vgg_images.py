from vgg_packages import *

A_TRAIN_DIR = 'path of train data'
A_VALID_DIR = 'path of valid data'
A_TEST_DIR = 'path of test data'

FIRE_TEST_DIR = 'path of fireblight data'
SCAB_TEST_DIR = 'path of scab data'
BLAC_TEST_DIR = 'path of blacknecroticleafspot data'
NORM_TEST_DIR = 'path of normal data'

train_file_list = os.listdir(A_TRAIN_DIR)
valid_file_list = os.listdir(A_VALID_DIR)
test_file_list = os.listdir(A_TEST_DIR)

fire_file_list = os.listdir(FIRE_TEST_DIR)
scab_file_list = os.listdir(SCAB_TEST_DIR)
blac_file_list = os.listdir(BLAC_TEST_DIR)
norm_file_list = os.listdir(NORM_TEST_DIR)

IMG_SIZE = 224

def img_label(lbl):
    disease_label = lbl.split('_')[0] + "_" + lbl.split('_')[1]

    # 배
    if disease_label == 'pear_fireblight':
        return [1, 0, 0, 0] # 0
    elif disease_label == 'pear_scab':
        return [0, 1, 0, 0] # 1
    elif disease_label == 'pear_blacknecroticleafspot':
        return [0, 0, 1, 0] # 2
    elif disease_label == 'pear_normal':
        return [0, 0, 0, 1] # 3

#########################################################

def create_train_data():
    train_data = []

    for img in train_file_list:
        path = os.path.join(A_TRAIN_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        train_data.append(img)

    train_img_array = np.asarray(train_data)
    return train_img_array

def create_train_label():
    train_label = []

    for lbl in train_file_list:
        label = img_label(lbl)
        train_label.append(label)

    train_lbl_array = np.asarray(train_label)
    return train_lbl_array

#########################################################

def create_valid_data():
    valid_data = []

    for img in valid_file_list:
        path = os.path.join(A_VALID_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        valid_data.append(img)

    valid_img_array = np.asarray(valid_data)
    return valid_img_array

def create_valid_label():
    valid_label = []

    for lbl in valid_file_list:
        label = img_label(lbl)
        valid_label.append(label)

    valid_lbl_array = np.asarray(valid_label)
    return valid_lbl_array

#########################################################

def create_test_data():
    test_data = []

    for img in test_file_list:
        path = os.path.join(A_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_data.append(img)

    test_img_array = np.asarray(test_data)
    return test_img_array

def create_test_label():
    test_label = []

    for lbl in test_file_list:
        label = img_label(lbl)
        test_label.append(label)

    test_lbl_array = np.asarray(test_label)
    return test_lbl_array

#########################################################

def create_fire_data():
    fire_data = []

    for img in fire_file_list:
        path = os.path.join(FIRE_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        fire_data.append(img)

    fire_img_array = np.asarray(fire_data)
    return fire_img_array

def create_fire_label():
    fire_label = []

    for lbl in fire_file_list:
        label = img_label(lbl)
        fire_label.append(label)

    fire_lbl_array = np.asarray(fire_label)
    return fire_lbl_array

#########################################################

def create_scab_data():
    scab_data = []

    for img in scab_file_list:
        path = os.path.join(SCAB_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        scab_data.append(img)

    scab_img_array = np.asarray(scab_data)
    return scab_img_array

def create_scab_label():
    scab_label = []

    for lbl in scab_file_list:
        label = img_label(lbl)
        scab_label.append(label)

    scab_lbl_array = np.asarray(scab_label)
    return scab_lbl_array

########################################################

def create_blac_data():
    blac_data = []

    for img in blac_file_list:
        path = os.path.join(BLAC_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        blac_data.append(img)

    blac_img_array = np.asarray(blac_data)
    return blac_img_array

def create_blac_label():
    blac_label = []

    for lbl in blac_file_list:
        label = img_label(lbl)
        blac_label.append(label)

    blac_lbl_array = np.asarray(blac_label)
    return blac_lbl_array

########################################################

def create_norm_data():
    norm_data = []

    for img in norm_file_list:
        path = os.path.join(NORM_TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        norm_data.append(img)

    norm_img_array = np.asarray(norm_data)
    return norm_img_array

def create_norm_label():
    norm_label = []

    for lbl in norm_file_list:
        label = img_label(lbl)
        norm_label.append(label)

    norm_lbl_array = np.asarray(norm_label)
    return norm_lbl_array