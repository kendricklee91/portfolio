# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
from sanic import Blueprint, response
import utility as util
from model.fm.tasks import inference_task
from settings import ENVS
import messages

api_bp = Blueprint('fm_api')


@api_bp.route(ENVS['AI_FM_URL'], methods=['POST'])
async def fm_api(request):
    """
    FM API

    Parameters
    ----------
    request: sanic.request.Request
        Sanic Request Object

    Returns
    -------
    sanic.response.HTTPResponse
        Sanic Response Object
    """
    validation_rule = request.app.config.AI_FM_SETTINGS['REQUEST_VALIDATION_RULE']

    params, errs = util.validate(validation_rule, request.json)
    if errs:
        params['errs'] = errs
        return response.json(params, status=400)

    inference_task.delay(params, request.app.config.AI_FM_SETTINGS, request.app.config.AI_FM_RULES)  # 비동기 테스크

    params['result'] = True
    params['msg'] = messages.MSG_JOB_STARTED
    return response.json(params)
