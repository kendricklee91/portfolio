# 2020-05-09 Bokyong Seo<kay.seo@storypot.io>

queries = {
    # 데이터 읽어오기
    'read': {
        'jindan': '''
            SELECT
                JH.QuizCode,
                JH.UserID,
                JH.Correct
            FROM HBEdu_App.DBO.TBL_APP_HBMath_Jindan_His AS JH
                LEFT OUTER JOIN HBEdu_App.dbo.TBL_APP_HBMath_JindanReset_His AS JRH
                    ON JH.UserID = JRH.UserID AND JH.GroupID = JRH.GroupID
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS TPA
                    ON JH.QuizCode = TPA.f_problem_id
            WHERE TPA.f_area_cd = 'MA'
                AND JH.CreDate > dateadd(mm, %(months)s, datediff(d, 0, getdate()))
        ''',  # 진단평가

        'subject': '''
            SELECT
                THD.QuizCode,
                THD.UserID,
                THD.Correct
            FROM HBEdu_App.dbo.TBL_APP_TestHisDtl AS THD
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS TPA
                    ON THD.QuizCode = TPA.f_problem_id
            WHERE TPA.f_area_cd = 'MA'
                AND THD.CreDate > dateadd(mm, %(months)s, datediff(d, 0, getdate()))
        ''',  # 과목별 학교 공부

        'achive': '''
            SELECT
                ATH.QuizCode,
                ATH.UserID,
                ATH.Correct
            FROM HBEdu_App.dbo.TBL_APP_AchieveTest_His AS ATH
                LEFT OUTER JOIN HBEdu_App.dbo.TBL_APP_AchieveTest_Log AS ATL
                    ON ATH.UserID = ATL.UserID AND ATH.GroupID = ATL.GroupID
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS TPA
                    ON ATH.QuizCode = TPA.f_problem_id AND TPA.f_area_cd = 'MA'
            WHERE ATH.CreDate > dateadd(mm, %(months)s, datediff(d, 0, getdate()))
        ''',  # 성취도 평가

        'bogang': '''
            SELECT
                BTHD.QuizCode,
                BTHD.UserID,
                BTHD.Correct
            FROM HBEdu_App.dbo.TBL_APP_BogangTestHisDtl AS BTHD
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS TPA
                    ON BTHD.QuizCode = TPA.f_problem_id
                LEFT OUTER JOIN HBEdu_App.dbo.TBL_APP_BogangTestHis AS BTH
                    ON BTHD.UserId = BTH.UserId AND BTHD.mCode = BTH.mCode
            WHERE TPA.f_area_cd = 'MA'
                AND BTHD.Answer is Not NULL
                AND BTH.CreDate > dateadd(mm, %(months)s, datediff(d, 0, getdate()))
        ''',  # 보충, 심화

        'twin': '''
            SELECT
                STD.QuizCode,
                STD.UserID,
                STD.Correct
            FROM HBEdu_App.dbo.TBL_APP_SimilarTestHisDtl AS STD
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS TPA
                    ON STD.QuizCode = TPA.f_problem_id
            WHERE TPA.f_area_cd = 'MA'
                AND STD.CreDate > dateadd(mm, %(months)s, datediff(d, 0, getdate()))
        ''',  # 쌍둥이 문제
    },

    # 데이터 쓰기
    'write': {
        'item': '''
            INSERT INTO irt_problem (quizcode, difficulty, difficulty_code, discrimination, discrimination_code)
                VALUES (%(item_id)s, %(item_difficulty)s, %(item_difficulty_code)s, %(item_discrimination)s, %(item_discrimination_code)s)
        ''',  # 문항 난이도, 변별도

        'student': '''
            INSERT INTO irt_student (userid, ability)
                VALUES (%(student_id)s, %(student_ability)s)
        ''',  # 학습자 능력
    }

}


def get_query(query_name, param):
    return queries[param][query_name]
