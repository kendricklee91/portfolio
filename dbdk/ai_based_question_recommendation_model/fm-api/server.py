# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sentry_sdk
from sentry_sdk.integrations.sanic import SanicIntegration
from sanic import Sanic
from api import api_bp
from db.connector import DBConnector
import settings
from settings import ENVS
import rules


def main():
    """
    Sanic 서버 시작점

    Returns
    -------
    None
    """
    sentry_sdk.init(
        dsn="http://ec72d7c66e224c7bbeba12023c039d86@183.110.210.106:9000/2",
        integrations=[SanicIntegration()],
        environment='-'.join(ENVS['AI_FM_URL'].split('/')[1:])
    )  # 센트리 초기화

    service_url = ENVS['AI_FM_URL']
    service_type = ENVS['AI_FM_TYPE']

    db_info = ENVS['AI_FM_DB']
    db_connector = DBConnector(**db_info)
    s = settings.load(service_url, service_type, db_connector)
    s.update(ENVS)
    r = rules.load(service_url, service_type, db_connector)

    app = Sanic(ENVS['AI_FM_URL'])
    app.update_config(s)

    app.config.AI_FM_SETTINGS = s
    app.config.AI_FM_RULES = r

    app.blueprint(api_bp)

    is_debug = True if ENVS['AI_FM_SANIC_MODE'].upper() == 'DEBUG' else False

    app.run(host=s['API_FM_HOST'], port=s['API_FM_PORT'],
            workers=s['SANIC_WORKERS'], debug=is_debug, access_log=is_debug)


if __name__ == "__main__":
    main()
