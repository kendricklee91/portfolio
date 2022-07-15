# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import tensorflow as tf
from model.dkt import metrics as mtrc
from model.dkt.datautil import get_target
import settings


class DKT:
    """
    DKT 모델 학습

    Attributes
    ----------
    train_data: tf.data.Dataset
        Train dataset
    valid_data: tf.data.Dataset
        Validation dataset
    test_data: tf.data.Dataset
        Test dataset
    nb_features: int
        Features depth
    nb_skills: int
        Skill depth
    num_of_gpus: int
        선택된 GPU의 수
    result_model: tf.keras.Model
        DKT Model
    result_eval: list
        Loss value & metrics values

    Methods
    -------
    train()
        환경(GPU/CPU)에 따른 DKT 모델 학습 및 평가 진행
    _train_model()
        DKT 모델 레이어 구성 및 모델 컴파일
    """

    def __init__(self, course, skill, train_data, valid_data, test_data, skill_map, num_of_gpus):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.nb_features = skill_map['features_depth']
        self.nb_skills = skill_map['skill_depth']

        self.num_of_gpus = num_of_gpus

        self.learning_rate = settings.BATCH_OPTIONS[course][skill]['learning_rate']
        self.hidden_units = settings.BATCH_OPTIONS[course][skill]['hidden_units']
        self.dropout_rate = settings.BATCH_OPTIONS[course][skill]['dropout_rate']
        self.optimizer = settings.BATCH_OPTIONS[course][skill]['optimizer']
        self.callbacks = settings.BATCH_OPTIONS[course][skill]['callbacks']
        self.shuffle = settings.BATCH_OPTIONS[course][skill]['shuffle']
        self.verbose = settings.BATCH_OPTIONS[course][skill]['verbose']
        self.epochs = settings.BATCH_OPTIONS[course][skill]['epochs']

        self.result_model = None
        self.result_eval = None

    def train(self):
        if self.num_of_gpus > 1:
            with tf.distribute.MirroredStrategy().scope():
                self._train_model()
        else:
            self._train_model()

        self.result_eval = self.result_model.evaluate(self.test_data, verbose=self.verbose, steps=None, callbacks=None)

    def _train_model(self):
        def custom_loss(y_true, y_pred):
            y_true, y_pred = get_target(y_true, y_pred)
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Layers of lstm model
        inputs = tf.keras.Input(shape=(None, self.nb_features))

        x = tf.keras.layers.Masking(mask_value=settings.PREP_MASK_VALUE)(inputs)
        x = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate)(x)

        dense = tf.keras.layers.Dense(self.nb_skills, activation='sigmoid')
        outputs = tf.keras.layers.TimeDistributed(dense)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        model.compile(loss=custom_loss, optimizer=optimizer, metrics=[
                      mtrc.BinaryAccuracy(), mtrc.AUC(), mtrc.Precision(), mtrc.Recall()])

        _ = model.fit(x=self.train_data, epochs=self.epochs, verbose=self.verbose, callbacks=self.callbacks,
                      validation_data=self.valid_data, shuffle=self.shuffle, initial_epoch=0, steps_per_epoch=None,
                      validation_steps=None, validation_freq=1)

        self.result_model = model
