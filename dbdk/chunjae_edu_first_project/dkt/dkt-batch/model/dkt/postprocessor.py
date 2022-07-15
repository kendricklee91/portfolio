# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import os
import json
import subprocess
import tensorflow as tf
import messages
import settings


def save_model(course, skill, model, skill_map):
    """
    DKT 학습된 모델, Feature depth, Skill depth, Skill code와 factorized value의 매핑 정보를 파일로 저장하고,
    Git 레포지토리에 커밋 & 푸시

    Parameters
    ----------
    course: str
        트레이닝 할 코스. 매일학교공부-'thd' / 성취도평가-'ath'
    skill: str
        트레이닝 할 스킬
    model: tf.keras.Model
        DKT Model
    skill_map: dict
        스킬코드 매핑 정보 딕셔너리

    Returns
    -------
    None
    """
    if not os.path.isdir(settings.PERSISTENT_DIR):
        subprocess.run(['git', 'clone', settings.GIT_MODEL_REPO, settings.PERSISTENT_DIR])

    # DKT Model
    model_path = settings.MODEL_PATH.format(course, skill)
    tf.keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)

    # 스킬코드 매핑 정보 딕셔너리
    skill_map_path = settings.MODEL_SKILL_MAP_PATH.format(course, skill)
    with open(skill_map_path, 'w') as skill_map_file:
        json.dump(skill_map, skill_map_file)

    # Git commit & push
    subprocess.run([
        'bash',
        '-c',
        'cd {} && git add * && git commit -m "{}" && git push origin master'.format(
            settings.PERSISTENT_DIR, messages.MSG_GIT_COMMIT.format(course, skill))
    ])
