# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import pandas as pd


class PostProcessorWeak:
    """
    취약 판단 후처리
    """

    def __init__(self, df, params, ai_rules, db_connector):
        self.df = df
        self.subject = params['subject']
        self.service_type = params['service_type']
        self.call_datetime = params['call_datetime']
        self.mcode = params['mCode']
        self.params = params
        self.ai_rules = ai_rules
        self.db_connector = db_connector

    def postprocess(self):
        topic_df = self._postprocess_topic()
        quiz_df = self._postprocess_quiz()
        self._insert_data(topic_df, quiz_df)

    def _postprocess_topic(self):
        df = self.df.copy()
        df = df[['UserID', 'QuizCode', 'topiccode', 'map_ptype_cd', 'Correct', 'Prediction']]
        df.loc[:, 'UserID'] = df.loc[:, 'UserID'].astype(str)
        df.loc[:, 'judge_mCode'] = self.mcode
        df.loc[df['Correct'] == -1, 'Solved'] = 'N'
        df.loc[df['Correct'] != -1, 'Solved'] = 'Y'

        group_list = []
        for key, group in df.groupby('topiccode'):
            ptypes = list(map(str.upper, set(group['map_ptype_cd'])))
            ptype_sums = []
            for ptype in ptypes:
                ptype_sums.append(
                    group[group['map_ptype_cd'] == ptype].shape[0] * self.ai_rules[f"WEAK_{self.service_type}_{ptype}AR_WEIGHT"])
                ptype_sums.append(
                    group[group['map_ptype_cd'] == ptype].shape[0] * self.ai_rules[f"WEAK_{self.service_type}_{ptype}IR_WEIGHT"])

            ptype_sum = sum(ptype_sums)

            def _calc_quiz_score(ptype, solved, correct, prediction):
                weight_ar = self.ai_rules[f"WEAK_{self.service_type}_{ptype}AR_WEIGHT"]
                weight_ir = self.ai_rules[f"WEAK_{self.service_type}_{ptype}IR_WEIGHT"]
                if solved == 'Y':
                    qs = (weight_ar * correct + weight_ir * prediction) / ptype_sum
                else:  # 'N'
                    qs = (weight_ar * prediction + weight_ir * prediction) / ptype_sum
                return qs

            group['quiz_score'] = group.apply(
                lambda row: _calc_quiz_score(row['map_ptype_cd'], row['Solved'], row['Correct'], row['Prediction']),
                axis=1)

            score = group['quiz_score'].sum()
            label = self._judge_label(score)

            group.loc[:, 'Score'] = score
            group.loc[:, 'StudyContentYN'] = label
            group_list.append(group)

        g_df = pd.concat(group_list)

        g_df = g_df[['UserID', 'judge_mCode', 'topiccode', 'Score', 'StudyContentYN']]
        g_df = g_df.drop_duplicates()
        g_df = g_df.reset_index(drop=True)
        g_df.loc[:, 'Subject'] = self.subject
        g_df.loc[:, 'ServiceType'] = self.service_type
        g_df.loc[:, 'CallDateTime'] = self.call_datetime
        return g_df

    def _judge_label(self, score):
        label = None
        score = score * 100
        if score >= self.ai_rules['WEAK_JUDGE_SCORE_HIGH']:  # 취약 개념 x, 취약 문항 x
            label = self.ai_rules['WEAK_JUDGE_LABEL_HIGH']
        elif self.ai_rules['WEAK_JUDGE_SCORE_HIGH'] > score >= self.ai_rules[
            'WEAK_JUDGE_SCORE_LOW']:  # 취약 개념 x, 취약 문항 o
            label = self.ai_rules['WEAK_JUDGE_LABEL_MID']
        elif self.ai_rules['WEAK_JUDGE_SCORE_LOW'] > score:  # 취약 개념 o, 취약 문항 o
            label = self.ai_rules['WEAK_JUDGE_LABEL_LOW']

        return label

    def _postprocess_quiz(self):
        df = self.df.copy()

        # 0 = 풀이하지 않은 경우, 1 = 풀이한 경우
        df.loc[df['Correct'] == -1, 'Solved'] = 0
        df.loc[df['Correct'] != -1, 'Solved'] = 1

        df['Correct'] = df['Correct'].replace(-1, '')  # 풀이하지 않은 경우 정오답 정보는 빈 값으로

        df = df[['topiccode', 'map_ptype_cd', 'QuizCode', 'Solved', 'Prediction', 'Correct', 'mCode']]

        return df

    def _insert_data(self, topic_df, quiz_df):
        topic_list = topic_df.values.tolist()
        with self.db_connector as connected:
            topic_query = connected.get_query('AiKnowledgeMap_APIResultTopicList', 'create')
            quiz_query = connected.get_query('AiKnowledgeMap_APIResultQuizList', 'create')
            with connected.conn.cursor() as cursor:
                for topic_data in topic_list:
                    cursor.execute(topic_query, topic_data)
                    topic_id = cursor.fetchone()[0]
                    topic_quiz_df = quiz_df[quiz_df['topiccode'] == topic_data[2]]
                    topic_quiz_df['Topic_ID'] = topic_id
                    topic_quiz_df = topic_quiz_df[['Topic_ID', 'map_ptype_cd', 'QuizCode', 'Solved', 'Prediction', 'Correct', 'mCode']]
                    cursor.executemany(quiz_query, topic_quiz_df.values.tolist())
