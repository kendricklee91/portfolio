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
        'M': '''
            SELECT 
                thd.UserID,
                thd.mCode,
                thd.QuizCode,
                CAST(CASE WHEN thd.correct='O' THEN 1 ELSE 0 END AS INT) AS Correct,
                aikmap.topiccode,
                tpa.f_analysis_id AS cognitive_cd,
                tpa.f_studytree_id AS content_cd,
                tp.f_ptype_cd AS ptype_cd,  -- 문제 답변형태 코드
                prev_kc.n1_curr_kc_cd,  -- 1 이전 지식 개념 코드
                prev_kc.n1_curr_kc_correct_ratio,  -- 1 이전 지식 개념 정답률
                prev_kc.n2_curr_kc_cd,  -- 2 이전 지식 개념 코드
                prev_kc.n2_curr_kc_correct_ratio,  -- 2 이전 지식 개념 정답률
                prev_kc.n3_curr_kc_cd,  -- 3 이전 지식 개념 코드
                prev_kc.n3_curr_kc_correct_ratio,  -- 3 이전 지식 개념 정답률
                irt.difficulty_eval,
                irt.discrimination_eval
            FROM HBEdu_App.dbo.TBL_APP_TestHisDtl AS thd
                LEFT JOIN ( 
                    SELECT 
                        aiqc.lt_code,
                        aiqc.difficulty_eval AS difficulty_eval,
                        aiqc.discrimination_eval AS discrimination_eval
                    FROM
                        HBEdu_App_Study.dbo.AiKnowledgeMap_IRTQuizCode AS aiqc
                    ) AS irt on irt.lt_code = thd.QuizCode
                LEFT JOIN (
                    SELECT
                    mqt.lt_code,
                    mqt.topiccode
                    FROM
                        HBEdu_App_Study.dbo.AiKnowledgeMap_QuizCodeTopic AS mqt
                        INNER JOIN 
                            HBEdu_App_Study.dbo.AiKnowledgeMap_Topic AS kmapTopic
                            ON mqt.topiccode = kmapTopic.topiccode AND kmapTopic.useyn = 1
                    ) AS aikmap ON aikmap.lt_code = thd.QuizCode
                /* 이전 지식 개념 */
                LEFT JOIN (
                    SELECT
                        a.userid, a.mcode,
                        LAG(a.mcode, 1) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n1_curr_kc_cd,
                        LAG(a.correct_ratio, 1) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n1_curr_kc_correct_ratio,
                        LAG(a.mcode, 2) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n2_curr_kc_cd,
                        LAG(a.correct_ratio, 2) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n2_curr_kc_correct_ratio,
                        LAG(a.mcode, 3) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n3_curr_kc_Cd,
                        LAG(a.correct_ratio, 3) OVER (PARTITION BY a.userid, a.lec ORDER BY a.L_IDX) AS n3_curr_kc_correct_ratio
                    FROM (
                        SELECT
                            a.UserID,
                            a.mCode,
                            LEFT(a.mCode, 9) AS lec,
                            b.L_IDX,
                            AVG(CAST(CASE WHEN a.correct = 'O' THEN 1 ELSE 0 END AS FLOAT)) AS correct_ratio
                        FROM HBEdu_App.dbo.TBL_APP_TestHisDtl AS a
                            LEFT JOIN HBEdu_Hbstudy.dbo.TBL_LECTURE_LCMS AS b
                                ON a.mCode = b.MCode
                        GROUP BY a.UserID, a.mCode, b.L_IDX
                        ) AS a
                        LEFT JOIN HBEdu_Hbstudy.dbo.TBL_LECTURE_LCMS AS b
                            ON a.mCode = b.mCode
                        LEFT JOIN HBEdu_Hbstudy.dbo.TBL_UNIT_LCMS AS c
                            ON b.LecCode = c.LecCode
                    WHERE c.subject = %(subject)s
                    ) AS prev_kc
                    ON thd.UserID = prev_kc.UserID AND thd.mCode = prev_kc.mCode
                JOIN HBEdu_Hbstudy.dbo.TBL_LECTURE_LCMS AS llcms
                    ON thd.mCode = llcms.MCode
                JOIN HBEdu_Hbstudy.dbo.TBL_UNIT_LCMS AS ulcms
                    ON llcms.LecCode = ulcms.LecCode
                LEFT JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS tpa
                    ON thd.QuizCode = tpa.f_problem_id 
                LEFT JOIN HBEdu_App_Edubase.dbo.t_problem AS tp
                    ON tpa.f_file_id = tp.f_file_id
                LEFT JOIN HBEdu_App_Edubase.dbo.t_code AS tc
                    ON tp.f_ptype_cd = tc.f_detail_id
                LEFT JOIN HBEdu_App_Edubase.dbo.t_studytree AS tst1
                    ON tpa.f_analysis_id = tst1.f_studytree_id
                LEFT JOIN HBEdu_App_Edubase.dbo.t_studytree AS tst2
                    ON tpa.f_studytree_id = tst2.f_studytree_id
                LEFT JOIN HBEdu_Passport.dbo.Membership_MembershipInfomation AS mem_info
                    ON thd.UserID = mem_info.UserId
            WHERE tpa.f_eduprocess_cd IN ('01', '09', '9A')
                AND ulcms.subject = ?
                AND tpa.f_deleteyn = 'N' 
                AND mem_info.TestUserIDYN <> 'Y'
        ''',  # 수학
        'KESN': '''
            SELECT
                thd.UserID,
                thd.mCode,
                thd.QuizCode,
                CAST(CASE WHEN thd.correct='O' THEN 1 ELSE 0 END AS INT) AS Correct,
                aikmap.topiccode,
                tp.f_ptype_cd AS ptype_cd,  -- 문제 답변형태 코드
                irt.difficulty_eval,
                irt.discrimination_eval
            FROM HBEdu_App.dbo.TBL_APP_TestHisDtl AS thd
                LEFT JOIN (
                    SELECT 
                        aiqc.lt_code,
                        aiqc.difficulty_eval AS difficulty_eval,
                        aiqc.discrimination_eval AS discrimination_eval
                    FROM
                        HBEdu_App_Study.dbo.AiKnowledgeMap_IRTQuizCode AS aiqc
                    ) AS irt
                    ON irt.lt_code = thd.QuizCode
                LEFT JOIN (
                    SELECT
                        mqt.lt_code,
                        mqt.topiccode
                    FROM HBEdu_App_Study.dbo.AiKnowledgeMap_QuizCodeTopic AS mqt
                        INNER JOIN HBEdu_App_Study.dbo.AiKnowledgeMap_Topic AS kmapTopic
                            ON mqt.topiccode = kmapTopic.topiccode AND kmapTopic.useyn = 1
                    ) AS aikmap
                    ON aikmap.lt_code = thd.QuizCode
                JOIN HBEdu_Hbstudy.dbo.TBL_LECTURE_LCMS AS llcms
                    ON thd.mCode = llcms.MCode
                JOIN HBEdu_Hbstudy.dbo.TBL_UNIT_LCMS AS ulcms
                    ON llcms.LecCode = ulcms.LecCode
                LEFT JOIN HBEdu_App_Edubase.dbo.t_problem_analysis AS tpa
                    ON thd.QuizCode = tpa.f_problem_id 
                LEFT JOIN HBEdu_App_Edubase.dbo.t_problem AS tp
                    ON tpa.f_file_id = tp.f_file_id
                LEFT JOIN HBEdu_App_Edubase.dbo.t_code AS tc
                    ON tp.f_ptype_cd = tc.f_detail_id
                LEFT JOIN HBEdu_Passport.dbo.Membership_MembershipInfomation AS mem_info
                    ON thd.UserID = mem_info.UserId
            WHERE tpa.f_eduprocess_cd IN ('01', '09', '9A')
                AND ulcms.subject = ?
                AND tpa.f_deleteyn = 'N' 
                AND mem_info.TestUserIDYN <> 'Y'
        '''  # 국어, 영어, 과학, 사회
    }
}
