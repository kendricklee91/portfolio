# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sentry_sdk
from worker import celery_app
from model.fm import controller


@celery_app.task
def inference_task(params, ai_settings, ai_rules):
    """
    Celery FM 추론 작업

    Parameters
    ----------
    params: dict
        API 요청 파라메터
    ai_settings: dict
        DB의 API 설정
    ai_rules: dict
        DB의 인공지능 룰셋

    Returns
    -------
    None
    """
    try:
        sentry_sdk.set_tag("userid", params['userid'])
        sentry_sdk.set_tag("mCode", params['mCode'])
        controller.inference(params, ai_settings, ai_rules)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise e
