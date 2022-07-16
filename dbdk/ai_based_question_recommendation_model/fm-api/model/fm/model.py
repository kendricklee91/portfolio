# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import os
import xlearn


def predict(df, input_filepath, model_path):
    """
    FM 모델 추론

    Parameters
    ----------
    df: pandas.Dataframe
        문항 데이터
    input_filepath: str
        문항 데이터의 ffm 파일 경로
    model_path: str
        모델 디렉토리 경로

    Returns
    -------
    pandas.Dataframe
        추론값이 포함된 문항 데이터
    """
    model = xlearn.create_ffm()
    model.setTest(input_filepath)
    model.setSigmoid()
    model.setQuiet()
    pred = model.predict(os.path.join(model_path, 'model_dm.out'))

    del model

    df['Prediction'] = pred

    return df
