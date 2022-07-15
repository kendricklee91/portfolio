# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import os
import sentry_sdk
from sentry_sdk.integrations.sanic import SanicIntegration
from sanic import Sanic
import subprocess
import json
from api import api_bp
import settings


def main():
    sentry_sdk.init(dsn=settings.SENTRY_DSN, integrations=[SanicIntegration()])  # 센트리 초기화

    app = Sanic('dkt_api')

    # Sanic 서버 설정
    app.config.REQUEST_MAX_SIZE = settings.REQUEST_MAX_SIZE
    app.config.REQUEST_TIMEOUT = settings.REQUEST_TIMEOUT
    app.config.RESPONSE_TIMEOUT = settings.RESPONSE_TIMEOUT

    app.config.MODEL_PARAMS = _load_model_files()  # Load model parameters from Git repository

    app.blueprint(api_bp)

    app.run(host=settings.API_HOST, port=settings.API_PORT,
            workers=settings.SANIC_WORKERS, debug=False, access_log=False)


def _load_model_files():
    """
    Model parameter file을 Git repository로부터 가져와 메모리에 적재

    Parameters
    ----------
    None

    Returns
    -------
    dict
        각 Course, Skill 별로 Model parameter를 담고 있는 딕셔너리
        구조: course > skill > serving_uri, nb_features, nb_skills, factorval
    """
    if not os.path.isdir(settings.PERSISTENT_DIR):
        subprocess.run(['git', 'clone', settings.GIT_MODEL_REPO, settings.PERSISTENT_DIR])

    # Git pull & restart tensorflow-serving
    subprocess.run([
        'bash',
        '-c',
        'cd {} && git pull origin master && supervisorctl restart tensorflow-serving'.format(settings.PERSISTENT_DIR)
    ])

    model_dict = {}
    for course, skills in settings.SKILL_CODES.items():
        course_dict = {}
        for skill in skills:
            skill_map = _load_skill_map(settings.MODEL_SKILL_MAP_PATH.format(course, skill))
            skill_map['serving_uri'] = settings.MODEL_TS_URI.format(course, skill)
            course_dict[skill] = skill_map

        model_dict[course] = course_dict

    return model_dict


def _load_skill_map(filepath):
    """
    스킬 코드 매핑 정보를 {persistent_path}/{course}_{skill}_skill_map.json 파일에서 읽어와 딕셔너리로 반환

    Parameters
    ----------
    filepath: str
        Course, skill별 파일 경로

    Returns
    -------
    dict
        스킬 코드 매핑 정보 딕셔너리
    """
    with open(filepath, 'r') as json_file:
        json_data = json.load(json_file)

    return json_data


if __name__ == "__main__":
    main()
