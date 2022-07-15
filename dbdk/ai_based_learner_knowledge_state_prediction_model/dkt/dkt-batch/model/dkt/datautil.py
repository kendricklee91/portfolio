# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import tensorflow as tf
import settings


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, settings.PREP_MASK_VALUE), y_true.dtype)
    skills, y_true = tf.split(y_true * mask, num_or_size_splits=[-1, 1], axis=-1)

    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)  # Get predictions for each skill
    
    y_pred = tf.where(tf.math.is_nan(y_pred), 0., y_pred)  # precision 값이 너무 커져 NaN으로 가지 않도록 y_pred 를 0 으로 변경함

    return y_true, y_pred
