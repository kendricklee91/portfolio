# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
queries = {
    'read': {
        'SETTINGS': '''
            SELECT 
                st.SettingsJSON
            FROM
                HBEdu_Report.TBL_RS_SERVICE AS s
                    JOIN
                HBEdu_Report.TBL_RS_SETTINGS AS st ON s.ID = st.Service_ID
            WHERE
                s.ServiceURL = %s AND s.Type = %s
                    AND st.Enabled = '1'
            ORDER BY st.CreateDateTime DESC
            LIMIT 1
        ''',  # 설정 가져오기
        'HYPERPARAMETER': '''
            SELECT 
                hp.HyperParametersJSON
            FROM
                HBEdu_Report.TBL_RS_SERVICE AS s
                    JOIN
                HBEdu_Report.TBL_RS_HYPERPARAMETER AS hp ON s.ID = hp.Service_ID
            WHERE
                s.ServiceURL = %s AND s.Type = %s
                    AND hp.Enabled = '1'
            ORDER BY hp.CreateDateTime DESC
            LIMIT 1
        '''  # 하이퍼파라메터 가져오기
    }
}

