# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import pandas as pd
import sentry_sdk


class PostProcessorPrior:
    """
    선행 판단 후처리
    """

    def __init__(self, df, params, ai_rules, db_connector):
        self.df = df
        self.user_level = params['userLevel']
        self.my_timetable_date = params['myTimetableDate']
        self.call_datetime = params['call_datetime']
        self.mcode = params['mCode']
        self.params = params
        self.ai_rules = ai_rules
        self.db_connector = db_connector

    def postprocess(self):
        mcode_df = self._postprocess_mcode()
        quiz_df = self._postprocess_quiz()
        self._insert_data(mcode_df, quiz_df)

    def _postprocess_mcode(self):
        df = self.df.copy()
        df = df[['UserID', 'mCode', 'mmCode', 'QuizCode', 'checkType', 'Correct', 'Prediction']]
        df.loc[:, 'UserID'] = df.loc[:, 'UserID'].astype(str)
        df.loc[:, 'judge_mCode'] = self.mcode
        df.loc[df['Correct'] == -1, 'Solved'] = 'N'
        df.loc[df['Correct'] != -1, 'Solved'] = 'Y'

        group_list = []
        for key, group in df.groupby('judge_mCode'):
            ct_1_cnt = group[group['checkType'] == 1].shape[0]
            ct_sum = ct_1_cnt * self.ai_rules['PRIOR_SOLVED_S_AR_WEIGHT']
            ct_sum += ct_1_cnt * self.ai_rules['PRIOR_SOLVED_S_IR_WEIGHT']

            ct_2_cnt = group[group['checkType'] == 2].shape[0]
            ct_sum += ct_2_cnt * self.ai_rules['PRIOR_SOLVED_U_IR_WEIGHT']

            def _calc_quiz_score(checkType, solved, correct, prediction):
                if checkType == 1:
                    weight_ar = self.ai_rules['PRIOR_SOLVED_S_AR_WEIGHT']
                    weight_ir = self.ai_rules['PRIOR_SOLVED_S_IR_WEIGHT']
                else:  # 2
                    if solved == 'Y':
                        weight_ar = self.ai_rules['PRIOR_SOLVED_U_IR_WEIGHT']
                        weight_ir = 0
                    else:
                        weight_ar = 0
                        weight_ir = self.ai_rules['PRIOR_SOLVED_U_IR_WEIGHT']

                if solved == 'Y':
                    score = (weight_ar * correct + weight_ir * prediction) / ct_sum
                else:  # 'N'
                    score = (weight_ar * prediction + weight_ir * prediction) / ct_sum
                return score

            group['quiz_score'] = group.apply(
                lambda row: _calc_quiz_score(row['checkType'], row['Solved'], row['Correct'], row['Prediction']),
                axis=1)

            score = group['quiz_score'].sum()
            label = self._judge_label(score)

            group.loc[:, 'Score'] = score
            group.loc[:, 'userLevel'] = label
            group_list.append(group)

        g_df = pd.concat(group_list)

        g_df = g_df[['UserID', 'judge_mCode', 'Score', 'userLevel']]
        g_df = g_df.drop_duplicates()
        g_df = g_df.reset_index(drop=True)

        g_df.loc[:, 'currentLevel'] = self.user_level
        g_df.loc[:, 'myTimetableDate'] = self.my_timetable_date
        g_df.loc[:, 'CallDateTime'] = self.call_datetime
        return g_df

    def _judge_label(self, score):
        label = None
        score = score * 100
        if score >= self.ai_rules['PRIOR_JUDGE_SCORE_HIGH']:
            label = self.ai_rules['PRIOR_JUDGE_LABEL_TOP']  # 최상
        elif self.ai_rules['PRIOR_JUDGE_SCORE_HIGH'] > score >= self.ai_rules['PRIOR_JUDGE_SCORE_MID']:
            label = self.ai_rules['PRIOR_JUDGE_LABEL_HIGH']  # 상
        elif self.ai_rules['PRIOR_JUDGE_SCORE_MID'] > score >= self.ai_rules['PRIOR_JUDGE_SCORE_LOW']:
            label = self.ai_rules['PRIOR_JUDGE_LABEL_MID']  # 중
        elif self.ai_rules['PRIOR_JUDGE_SCORE_LOW'] > score:
            label = self.ai_rules['PRIOR_JUDGE_LABEL_LOW']  # 하
        return label

    def _postprocess_quiz(self):
        df = self.df.copy()
        df.loc[:, 'judge_mCode'] = self.mcode

        # 0 = 풀이하지 않은 경우, 1 = 풀이한 경우
        df.loc[df['Correct'] == -1, 'Solved'] = 0
        df.loc[df['Correct'] != -1, 'Solved'] = 1

        df['Correct'] = df['Correct'].replace(-1, '')  # 풀이하지 않은 경우 정오답 정보는 빈 값으로

        df.loc[:, 'CallDateTime'] = self.call_datetime
        df = df[['judge_mCode', 'mmCode', 'QuizCode', 'Solved', 'Prediction', 'Correct', 'CallDateTime']]

        return df

    def _insert_data(self, mcode_df, quiz_df):
        mcode_list = mcode_df.values.tolist()
        with self.db_connector as connected:
            mcode_query = connected.get_query('AiPriorRecommend_APIResultmCodeList', 'create')
            quiz_query = connected.get_query('AiPriorRecommend_APIResultQuizList', 'create')
            with connected.conn.cursor() as cursor:
                for mcode_data in mcode_list:
                    cursor.execute(mcode_query, mcode_data)
                    mcode_id = cursor.fetchone()[0]
                    mcode_quiz_df = quiz_df[quiz_df['judge_mCode'] == mcode_data[1]]
                    mcode_quiz_df['mCode_IDX'] = mcode_id
                    mcode_quiz_df = mcode_quiz_df[
                        ['mCode_IDX', 'mmCode', 'QuizCode', 'Solved', 'Prediction', 'Correct', 'CallDateTime']]
                    cursor.executemany(quiz_query, mcode_quiz_df.values.tolist())
