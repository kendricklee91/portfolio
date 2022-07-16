# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
queries = {
    'read': {
        'AiService_Module_Settings': '''
            SELECT 
                st.SettingsJSON
            FROM
                TBL_RS_SERVICE AS s
                JOIN TBL_RS_SETTINGS AS st
                    ON s.ID = st.Service_ID
            WHERE
                s.ServiceURL = %s AND s.Type = %s
                    AND st.Enabled = 'Y'
            ORDER BY st.CreateDateTime DESC
            LIMIT 1;
        ''',
        'AiService_Module_HyperParameter': '''
            SELECT 
                hp.HyperParametersJSON
            FROM
                TBL_RS_SERVICE AS s
                JOIN TBL_RS_HYPERPARAMETER AS hp
                    ON s.ID = hp.Service_ID
            WHERE
                s.ServiceURL = %s AND s.Type = %s
                    AND hp.Enabled = 'Y'
            ORDER BY hp.CreateDateTime DESC
            LIMIT 1;
        '''
    }
}
