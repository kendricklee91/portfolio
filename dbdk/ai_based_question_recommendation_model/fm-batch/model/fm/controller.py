# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sentry_sdk
from model.fm import dataloader, preprocessor, model, postprocessor
import messages


def batch(ai_settings, ai_rules, db_connector):
    """
    FM Batch Controller

    Parameters
    ----------
    ai_settings: dict
        DB의 배치 설정
    ai_rules: dict
        DB의 인공지능 룰셋
    db_connector: DBConnector

    Returns
    -------
    None
    """
    subject = ai_settings['BATCH_FM_SUBJECT']
    xlearn_input_path = ai_settings['XLEARN_INPUT_PATH']
    xlearn_model_path = ai_settings['XLEARN_MODEL_PATH']

    df = dataloader.load_data(ai_rules['QUERY_NAME_TO_DATALOAD'], ai_rules['COLUMNS_TO_DATALOAD'], db_connector,
                              subject)

    train_data_path, test_data_path, feature_mapping, userid_mean, quizcode_mean, mcode_mean = preprocessor.preprocess(
        df, xlearn_input_path, ai_rules['COLUMN_DEFINITIONS_FOR_PREPROCESSING'])

    auc = model.train(xlearn_model_path, train_data_path, test_data_path, ai_rules['MODEL_PARAMS'])
    sentry_sdk.capture_message(messages.MSG_MODEL_AUC.format(auc))

    postprocessor.postprocess(xlearn_model_path, feature_mapping, userid_mean, quizcode_mean, mcode_mean)
