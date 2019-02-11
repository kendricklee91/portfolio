from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import AveragePooling2D

from keras_tqdm import TQDMNotebookCallback
from keras.regularizers import l2

from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.callbacks import EarlyStopping

from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.layers import Input
from keras.layers import merge
from keras.models import Model

from PIL import Image
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import pyfastcopy
import datetime
import graphviz
import random
import pydot
import keras
import tqdm
import json
import cv2
import os