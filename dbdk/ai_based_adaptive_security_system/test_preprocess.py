from ksv_model.preprocess.conv_payload import KnownAttackPreprocess
import ksv_model.config.const as cst

import os
import pandas as pd
import unittest

class TestPreProcess(unittest.TestCase):

    # Fixture
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    @unittest.skip
    def test_sqli_preprocess(self):
        attack_type     = 'sqli'
        max_padd_size   = 1024
        load_file_dir   = os.path.join(cst.PATH_DATA, '{}_ps_3dayvalid_01.csv')
        save_file_dir   = os.path.join(cst.PATH_DATA, '{}_ps_3day1024padding_01.csv')
        
        kap             = KnownAttackPreprocess()
        
        df              = kap._get_data_df(load_file_dir, attack_type)
        payload_ps      = kap.preprocess(df, attack_type, max_padd_size)

        # save csv file
        save_file       = kap._get_save_df(payload_ps, save_file_dir, attack_type)


    def test_ig_preprocess(self):
        tdiff   = 120
        # nSIMS 데이터 파일 
        load_file_nsims         = os.path.join(cst.PATH_DATA, 'ig_nsims_20190831_01.csv')
        # LABEL 데이터 파일 
        load_file_label         = os.path.join(cst.PATH_DATA, 'ig_ps_20190830_label_01.csv')
        
        save_file_dir   = os.path.join(cst.PATH_DATA, 'ig_ps_20190831_inference_01.csv')

        kap             = KnownAttackPreprocess2()
        
        df              = kap._get_data_df(load_file_nsims)
        df_label        = kap._get_data_label_df(load_file_label)
        ig_ps           = kap.ig_preprocess(df, df_label, tdiff, type_="inference")
        #payload_ps     = kap.ig_preprocess(df,  tdiff, type_="inference")

        # save csv file
        save_file       = kap.get_save_df(ig_ps, save_file_dir)
if __name__ == "__main__":
    unittest.main()