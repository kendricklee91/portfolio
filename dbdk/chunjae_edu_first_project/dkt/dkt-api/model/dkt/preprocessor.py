# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import tensorflow as tf
import settings


def preprocess(course, skills, model_params, df):
    """
    각 Skill 별로 데이터 추출 및 전처리, 데이터셋 생성 작업을 거쳐 Tensorflow Serving Rest API에 예측 요청 가능한 List 형태로 변형

    Parameters
    ----------
    course: str
        코스. 단원마무리-'thd' / 성취도평가-'ath'
    skills: list
        스킬 리스트
    model_params: dict
        각 Course, Skill 별로 Model parameter를 담고 있는 딕셔너리
    df: pandas.DataFrame
        풀이이력 데이터 프레임

    Returns
    -------
    list
        전처리 데이터 딕셔너리의 리스트
        (특정 Course, Skill에 대한 데이터셋 및 Model parameter 등 정보를 포함한 딕셔너리의 리스트)
    """
    preprocessed_list = []
    for idx, skill in enumerate(skills):
        skill_df = _extract(course, skill, df)

        mp = model_params[skill]
        skill_df['skill_cd'] = skill_df['skill_cd'].astype(str)  # Skill code - Factorized value 매핑을 위한 string 캐스팅
        skill_df['skill'] = skill_df['skill_cd'].replace(mp['map_skill_factorval'])
        skill_df['skill_with_answer'] = skill_df['skill'] * 2 + skill_df['correct']

        if course == 'ath' and skill in ['difficulty', 'cognitive', 'content', 'ptype']:
            skill_df['skill_cd'] = skill_df['skill_cd'].replace(mp['map_skill_standard'])

        request_data = _generate_request_data(mp['features_depth'], mp['skill_depth'], skill_df)

        preprocessed_list.append({
            'skill': skill,
            'skill_idx': idx,
            'model_param': mp,
            'infer_df': skill_df,
            'request_data': request_data
        })

    return preprocessed_list


def _extract(course, skill, df):
    """
    각 Skill별 데이터 추출

    Parameters
    ----------
    course: str
        코스. 단원마무리-'thd' / 성취도평가-'ath'
    skill: str
        스킬
    df: pandas.DataFrame
        풀이이력 데이터 프레임

    Returns
    -------
    pandas.DataFrame
        각 Course, Skill 별로 추출한 풀이이력 데이터 프레임
    """

    skill_df = df[['user_id', skill, 'correct']].copy()
    skill_df.rename(columns={skill: 'skill_cd'}, inplace=True)

    # 마지막 문제까지 포함되도록 더미 데이터를 한 줄 추가
    nidx = len(skill_df)
    skill_df.loc[nidx] = skill_df.tail(1).values[0]
    skill_df.loc[nidx, 'correct'] = 1

    return skill_df


def _generate_request_data(nb_features, nb_skills, df):
    """
    전처리 된 데이터 프레임을 기반으로 데이터 셋을 생성하고, Tensorflow Serving Rest API에 예측 요청 가능한 리스트 형태로 변형

    Parameters
    ----------
    nb_features: int
        Feature depth
    nb_skills: int
        Skill depth
    df: pandas.DataFrame
        풀이이력 데이터 프레임

    Returns
    -------
    list
        Tensorflow Serving Rest API에 예측 요청 가능한 리스트
    """
    seq = df.groupby('user_id').apply(lambda r: (
        r['skill_with_answer'].values[:-1], r['skill'].values[1:], r['correct'].values[1:],))

    dataset = tf.data.Dataset.from_generator(generator=lambda: seq, output_types=(tf.int32, tf.int32, tf.float32))

    dataset = dataset.map(lambda feat, skill, label: (tf.one_hot(feat, depth=nb_features), tf.concat(
        values=[tf.one_hot(skill, depth=nb_skills), tf.expand_dims(label, -1)], axis=-1)))

    dataset = dataset.padded_batch(batch_size=1,
                                   padding_values=(settings.PREP_MASK_VALUE, settings.PREP_MASK_VALUE),
                                   padded_shapes=([None, None], [None, None]),
                                   drop_remainder=False)

    request_data = [t[0].numpy().tolist() for t in dataset][0]

    return request_data
