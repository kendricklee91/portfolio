# 2020-05-12 Bokyong Seo<gkbaturu@gmail.com>
import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from edward.models import Normal, Bernoulli, TransformedDistribution
import settings


class IRT:
    """
    IRT 모델 학습
    """

    def __init__(self):
        self._n_questions = None
        self._n_students = None

    def bayesian_inference(self, data, n_iter=settings.IRT_ITERATION, criticism=settings.IRT_CRITICISM):
        """
        추론을 실행하는 public mathod

        Parameters
        ----------
        data: pandas.DataFrame
        n_iter: int
        criticism: bool

        Returns
        -------
        pandas.DataFrame
            문항 난이도, 변별도 결과 데이터프레임
        pandas.DataFrame
            학생 능력 결과 데이터프레임
        """
        tf.reset_default_graph()  # Graph initialized
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())   # Variables initialized

        self._model(data)
        self._infer(n_iter)
        if criticism:
            self._criticism()

        output_item = {'item_id': range(self.n_questions), 'item_difficulty': self.q_b.mean().eval(),
                       'item_discrimination': tf.exp(self.q_a.distribution.mean()).eval()}
        output_student = {'student_id': range(self.n_students), 'student_ability': self.q_theta.mean().eval()}

        sess.close()

        return pd.DataFrame(output_item), pd.DataFrame(output_student)

    def _log_normal_q(self, shape, name=None):
        min_scale = settings.IRT_MIN_SCALE
        loc = tf.get_variable(f'{name}/loc', shape)
        scale = tf.get_variable(f'{name}/scale', shape,
                                initializer=tf.random_normal_initializer(stddev=settings.IRT_STDDEV))
        rv = TransformedDistribution(
            distribution=Normal(loc, tf.maximum(tf.nn.softplus(
                scale), min_scale), allow_nan_stats=False),
            bijector=tf.contrib.distributions.bijectors.Exp())
        return rv

    def _model(self, data):
        """
        사전분포 정의
        변별도(a), 난이도(b), 학습자 수준(theta) 모두 정규분포
        베르누이 Class를 통해 확률변수를 생성
        p = 1 / (1 + exp(-z))
        z = a * theta - b
        """
        self.obs = data.iloc[:, 2].values.astype(np.int32)
        self.student_ids = data.iloc[:, 1].values.astype(np.int32)
        self.question_ids = data.iloc[:, 0].values.astype(np.int32)

        self.n_students = len(set(self.student_ids))
        self.n_questions = len(set(self.question_ids))

        self.theta = Normal(loc=settings.IRT_ALPHA_MIN,
                            scale=settings.IRT_ALPHA_MAX, sample_shape=self.n_students)
        self.b = Normal(loc=settings.IRT_BETA_MIN,
                        scale=settings.IRT_BETA_MAX, sample_shape=self.n_questions)
        self.a = Normal(loc=settings.IRT_THETA_MIN,
                        scale=settings.IRT_THETA_MAX, sample_shape=self.n_questions)

        observation_logodds = tf.gather(self.a, self.question_ids) * (
            tf.gather(self.theta, self.student_ids) - tf.gather(self.b, self.question_ids))
        self.outcomes = Bernoulli(logits=observation_logodds)

    def _infer(self, n_iter=settings.IRT_ITERATION):
        """
        잠재 능력 추론
        추론은 Variation Inference 사용
        KL divergence를 통해 사전 분포에 가깝도록 최적화
        """
        self.q_theta = Normal(
            loc=tf.get_variable('q_theta/loc', [self.n_students]),
            scale=tf.nn.softplus(
                tf.get_variable('q_theta/scale', [self.n_students])))
        self.q_b = Normal(
            loc=tf.get_variable('q_b/loc', [self.n_questions]),
            scale=tf.nn.softplus(
                tf.get_variable('q_b/scale', [self.n_questions])))
        self.q_a = self._log_normal_q(self.n_questions, 'q_a')

        inference = ed.KLqp({self.theta: self.q_theta, self.b: self.q_b, self.a: self.q_a},
                            data={self.outcomes: self.obs})
        inference.run(n_iter=n_iter)

    def _criticism(self):
        """
        문항 변별도, 난이도와 학생 능력치를 그래프로 확인
        bayesian_inference 의 criticism 파라미터 True 설정 필요
        """
        color1 = '#3CAEA3'
        color2 = '#F6D55C'
        color3 = '#ED553B'

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.set_title('Distribution of Student Ability')
        ax2.set_title('Distribution of Item Difficulty')
        ax3.set_title('Distribution of Item Discrimination')
        ax1.set_xlabel('Student Ability')
        ax2.set_xlabel('Item Difficulty')
        ax3.set_xlabel('Item Discrimination')
        sns.distplot(self.q_theta.mean().eval(), ax=ax1, color=color1)
        sns.distplot(self.q_b.mean().eval(), ax=ax2, color=color2)
        sns.distplot(tf.exp(self.q_a.distribution.mean()).eval(), ax=ax3, color=color3)

        plt.savefig('irt_criticism.png')
        plt.show()
