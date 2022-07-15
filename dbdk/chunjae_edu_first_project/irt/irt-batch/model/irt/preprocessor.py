# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>
import os
import dask.dataframe as dd
import settings


def preprocess(df):
    """
    IRT 모델에 입력하기 위해 데이터 전처리 및 출력데이터로 변환할 매핑 리스트 생성

    Parameters
    ----------
    df: pandas.DataFrame
        배치에 사용할 풀이이력 데이터프레임

    Returns
    -------
    pandas.DataFrame
        전처리된 풀이이력 데이터프레임
    dict
        시퀀스와 문항ID의 매핑 테이블(출력데이터 변환용)
    dict
        시퀀스와 학생ID의 매핑 테이블(출력데이터 변환용)
    """
    npartitions = os.cpu_count() * settings.IRT_DASK_NP_MULTIPLIER
    ddf = dd.from_pandas(df, npartitions=npartitions)  # Use dask

    # Null 값 체크 - Correct 컬럼에 Null 이 있을 경우 사용하지 않음
    ddf.dropna(subset=['Correct'])

    # User ID를 UUID에서 String으로 변환
    ddf['UserID'] = ddf['UserID'].astype(str)

    # Correct 컬럼 값 변환 - X,O --> 0,1
    ddf['Correct'] = ddf['Correct'].replace('O', 1)
    ddf['Correct'] = ddf['Correct'].replace('X', 0)

    # 문항코드와 사용자 ID를 seq 로 변환
    cnt_item = sorted(df['QuizCode'].unique().tolist())
    cnt_id = sorted(df['UserID'].unique().tolist())

    # item to seq
    item_to_seq = {v: i for i, v in enumerate(cnt_item)}
    seq_to_item = {i: v for i, v in enumerate(cnt_item)}

    # id to seq
    student_to_seq = {v: i for i, v in enumerate(cnt_id)}
    seq_to_student = {i: v for i, v in enumerate(cnt_id)}

    ddf['QuizCode'] = ddf['QuizCode'].map(item_to_seq)
    ddf['UserID'] = ddf['UserID'].map(student_to_seq)

    result_df = ddf.compute()

    return result_df, seq_to_item, seq_to_student
