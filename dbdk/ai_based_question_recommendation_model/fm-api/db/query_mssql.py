# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
queries = {
    'read': {
        'AiService_Module_Settings': '''
            SELECT TOP 1
                st.SettingsJSON
            FROM
                AiService_Module_Service AS s
                JOIN AiService_Module_Settings AS st
                    ON s.ID = st.Service_ID
            WHERE s.ServiceURL = ?
                AND s.Type = ?
                AND st.Enabled = 1
            ORDER BY st.CreateDateTime DESC
        ''',  # 설정정보 가져오기
        'AiService_Module_HyperParameter': '''
            SELECT TOP 1
                hp.HyperParametersJSON
            FROM
                AiService_Module_Service AS s
                JOIN AiService_Module_HyperParameter AS hp
                    ON s.ID = hp.Service_ID
            WHERE s.ServiceURL = ?
                AND s.Type = ?
                AND hp.Enabled = 1
            ORDER BY hp.CreateDateTime DESC
        ''',  # 분석모델 메타정보 가져오기
        'USP_Quiz_Recommend_Weak': "EXEC USP_Quiz_Recommend_Weak '{userid}', '{mCode}', '{subject}'",
        'USP_Quiz_Recommend_KESS': "EXEC USP_Quiz_Recommend_KESS '{userid}', '{mCode}', '{subject}'",
        'USP_Quiz_Recommend_Pre': "EXEC USP_Quiz_Recommend_Pre '{userid}', '{mCode}', '{subject}', {userLevel}"
    },
    'create': {
        'AiKnowledgeMap_APIResultTopicList': '''
            INSERT INTO
                AiKnowledgeMap_APIResultTopicList (UserID, mCode, TopicCode, Score, StudyContentYN, Subject, ServiceType, CallDateTime)
            OUTPUT INSERTED.IDX
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        'AiKnowledgeMap_APIResultQuizList': '''
            INSERT INTO
                AiKnowledgeMap_APIResultQuizList (Topic_ID, QuizType, QuizCode, Solved, Predicted, Correct, TestHisMCode)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
        ''',
        'AiPriorRecommend_APIResultmCodeList': '''
            INSERT INTO
                AiPriorRecommend_APIResultmCodeList (UserID, mCode, Score, userLevel, currentLevel, myTimetableDate, CallDateTime)
            OUTPUT INSERTED.IDX
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
        ''',
        'AiPriorRecommend_APIResultQuizList': '''
            INSERT INTO
                AiPriorRecommend_APIResultQuizList (mCode_IDX, mmCode, QuizCode, Solved, Predicted, Correct, CallDateTime)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
        '''
    }
}
