# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>

queries = {
    'read': {
        'thd': '''
            SELECT
                LTRIM(RTRIM(thd.UserID)) AS user_id,
                LTRIM(RTRIM({})) AS skill_cd,
                LTRIM(RTRIM(ulcms.LecCode)) AS lesson_cd,
                LTRIM(RTRIM(ulcms.LecCode)) AS achieve_cd,
                LTRIM(RTRIM(thd.Correct)) AS correct
            FROM HBEdu_App.dbo.TBL_APP_TestHisDtl AS thd
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS tpa
                    ON thd.QuizCode = tpa.f_problem_id
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_problem AS tp
                    ON tpa.f_file_id = tp.f_file_id
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_code AS tc
                    ON tp.f_ptype_cd = tc.f_detail_id
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_studytree AS tst1
                    ON tpa.f_analysis_id = tst1.f_studytree_id
                LEFT OUTER JOIN HBEdu_App_Edubase.dbo.t_studytree AS tst2
                    ON tpa.f_studytree_id = tst2.f_studytree_id
                LEFT OUTER JOIN HBEdu_Hbstudy.dbo.TBL_LECTURE_LCMS AS llcms
                    ON thd.mCode = llcms.MCode
                LEFT OUTER JOIN HBEdu_Hbstudy.dbo.TBL_UNIT_LCMS AS ulcms
                    ON llcms.LecCode = ulcms.LecCode
            WHERE tpa.f_area_cd = 'MA'
                AND tc.f_group_id = '01'
                AND tst1.f_gubun ='02'
                AND tst1.f_eduprocess_cd IN ('9A', '09', '01')
                AND tst1.f_deleteyn = 'N'
                AND llcms.L_Type = 'T_EBOOK_B'
                AND ulcms.LecCourse = 'T0'
                AND thd.CreDate > dateadd(mm, %(months)s, dateadd(d, 1-datepart(weekday, getdate()), getdate()))
                AND thd.CreDate < dateadd(d, 1-datepart(weekday, getdate()), getdate())
                AND tst1.f_studytree_nm IN ('계산력', '문제해결력', '이해력', '추론력')
                AND tst2.f_studytree_nm IN ('규칙성', '도형', '수와 연산', '자료와 가능성', '측정')
            ORDER BY thd.UserID, ulcms.LecCode, llcms.MCode, thd.No, thd.CreDate
        ''',  # 매일학교공부

        'ath': '''
            SELECT
                LTRIM(RTRIM(ath.UserID)) AS user_id,
                LTRIM(RTRIM({})) AS skill_cd,
                LTRIM(RTRIM(atdm.LecCode)) AS lesson_cd,
                LTRIM(RTRIM(atdm.f_achieve_cd)) AS achieve_cd,
                LTRIM(RTRIM(ath.Correct)) AS correct
            FROM HBEdu_App.dbo.TBL_APP_AchieveTest_DKT_Mapping AS atdm
                LEFT OUTER JOIN HBEdu_App.dbo.TBL_APP_AchieveTest_His AS ath
                    ON atdm.QuizCode = ath.QuizCode
            WHERE
                ath.CreDate > dateadd(mm, %(months)s, dateadd(d, 1-datepart(weekday, getdate()), getdate()))
                AND ath.CreDate < dateadd(d, 1-datepart(weekday, getdate()), getdate())
            ORDER BY ath.UserID, atdm.QuizCode
        ''',  # 성취도 평가

        'map': '''
            SELECT
                LTRIM(RTRIM(skill_code)),
                LTRIM(RTRIM(standard_code)),
                factorval
            FROM HBEdu_App.dbo.TBL_APP_DKT_Mapping
            WHERE course = %(course)s
                AND skill_category = %(skill)s
                AND regdate = (SELECT MAX(regdate)
                                FROM HBEdu_App.dbo.TBL_APP_DKT_Mapping
                                WHERE course = %(course)s
                                    AND skill_category = %(skill)s)
            ORDER BY factorval
        ''',  # 매핑 테이블

        'map_max_factorval': '''
            SELECT
                MAX(factorval)
            FROM HBEdu_App.dbo.TBL_APP_DKT_Mapping
            WHERE course = %(course)s
                AND skill_category = %(skill)s
                AND regdate = (SELECT MAX(regdate)
                                FROM HBEdu_App.dbo.TBL_APP_DKT_Mapping
                                WHERE course = %(course)s
                                    AND skill_category = %(skill)s)
        '''  # Factorval 최대값
    }
}


def get_query(query_name, param):
    return queries[param][query_name]
