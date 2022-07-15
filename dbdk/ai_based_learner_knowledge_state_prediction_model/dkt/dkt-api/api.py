# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import re
from sanic import Blueprint, response
import utility as util
from model.dkt import controller
import settings

api_bp = Blueprint('dkt_inference', url_prefix='/dkt/inference')


@api_bp.route('/lecture', methods=['POST'])
async def dkt_inference_lecture(request):
    """
    단원마무리 DKT API

    Parameters
    ----------
    request: sanic.request.Request
        Sanic Request Object

    Returns
    -------
    sanic.response.HTTPResponse
        Sanic Response Object
    """
    errs = util.validate([
        {'param': 'userid', 'type': str},
        {'param': 'lesson', 'type': str},
        {'param': 'lectureList', 'type': list, 'child': [
            {'param': 'lecture', 'type': str},
            {'param': 'questionList', 'type': list, 'child': [
                {'param': 'cognitive', 'type': int},
                {'param': 'ptype', 'type': str},
                {'param': 'difficulty', 'type': str},
                {'param': 'correct', 'regex': re.compile('0|1')}
            ]}
        ]}
    ], request.json)

    if errs:
        return response.json({'errs': errs}, status=400)

    course = 'thd'
    skills = settings.SKILL_CODES[course]
    model_params = request.app.config['MODEL_PARAMS'][course]

    return response.json(controller.inference(course, skills, request.json, model_params))


@api_bp.route('/mat', methods=['POST'])
async def dkt_inference_mat(request):
    """
    성취도평가 DKT API

    Parameters
    ----------
    request: sanic.request.Request
        Sanic Request Object

    Returns
    -------
    sanic.response.HTTPResponse
        Sanic Response Object
    """
    errs = util.validate([
        {'param': 'userid', 'type': str},
        {'param': 'achieve', 'type': str},
        {'param': 'lessonList', 'type': list, 'child': [
            {'param': 'lesson', 'type': str},
            {'param': 'questionList', 'type': list, 'child': [
                {'param': 'cognitive', 'type': int},
                {'param': 'content', 'type': int},
                {'param': 'ptype', 'type': str},
                {'param': 'difficulty', 'type': str},
                {'param': 'correct', 'regex': re.compile('0|1')}
            ]}
        ]}
    ], request.json)

    if errs:
        return response.json({'errs': errs}, status=400)

    course = 'ath'
    skills = settings.SKILL_CODES[course]
    model_params = request.app.config['MODEL_PARAMS'][course]

    return response.json(controller.inference(course, skills, request.json, model_params))
