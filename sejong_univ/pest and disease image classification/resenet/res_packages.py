from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import ZeroPadding2D

from keras_tqdm import TQDMNotebookCallback
from keras.utils.data_utils import get_file
from keras.regularizers import l2
from keras.utils import layer_utils
from keras.utils import np_utils
from keras.utils.data_utils import get_file

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.engine.topology import get_source_inputs
from keras.utils.vis_utils import plot_model

from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.merge import add

from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import merge
from keras.models import Model

from keras import layers

import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import datetime
import keras
import tqdm
import json
import cv2
import six
import os