# -*- coding: utf-8 -*-
# set gpu dev

import os
import tensorflow  as tf
import keras.backend.tensorflow_backend as K
from keras.backend.tensorflow_backend import set_session


def set_dev_env():
    os.environ["CUDA_DEIVCE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_gpu_mem():
    # Set gpu memory limitation for multiple user
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def set_gpu_dev(gpu_dev):
    if gpu_dev is None:
        gpus = K._get_available_gpus()
        if not gpus:
            print('No available gpu device')    # critical
            return ''
        else:
            return gpus[0]
    else:
        return gpu_dev 
