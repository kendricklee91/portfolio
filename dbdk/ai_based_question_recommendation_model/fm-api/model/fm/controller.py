# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sys
from datetime import datetime
from db.connector import DBConnector
from model.fm import dataloader, preprocessor, model
from model.fm.postprocessors.weak import PostProcessorWeak
from model.fm.postprocessors.prior import PostProcessorPrior


def inference(params, ai_settings, ai_rules):
    """
    FM Inference Controller

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
    params.update({
        'subject': ai_settings['API_FM_SUBJECT'],
        'course': ai_settings['API_FM_COURSE'],
        'service_type': ai_settings['API_FM_SERVICE_TYPE'],
        'call_datetime': datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    })
    xlearn_input_path = ai_settings['XLEARN_INPUT_PATH']
    xlearn_model_path = ai_settings['XLEARN_MODEL_PATH']

    db_connector = DBConnector(**ai_settings['AI_FM_DB'])

    df, feature_mapping, userid_mean, quizcode_mean, mcode_mean = dataloader.load_data(params, xlearn_model_path,
                                                                                       ai_rules, db_connector)

    input_filepath = preprocessor.preprocess(df.copy(), feature_mapping, userid_mean, quizcode_mean, mcode_mean,
                                             xlearn_input_path, ai_rules['COLUMN_DEFINITIONS_FOR_PREPROCESSING'])

    result_df = model.predict(df, input_filepath, xlearn_model_path)

    postprocessor_cls = getattr(sys.modules[__name__], ai_rules['POSTPROCESSOR'])
    postprocessor = postprocessor_cls(result_df, params, ai_rules, db_connector)
    postprocessor.postprocess()
