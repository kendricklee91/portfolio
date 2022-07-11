import warnings
from ksv_model.periodic_process import mw_set_date, task_model_mw, task_model_dos
from ksv_model.model_mw import ModelMw
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
    def test_model_mw(self):
        mw = ModelMw()
        today, yesterday      = mw_set_date('2019-12-22 00:00:00')
        save_file_dir_type    = os.path.join(cst.PATH_DATA, 'mw_ps_type123_01.csv')
        #save_file_dir         = os.path.join(cst.PATH_DATA)

        # Malware Type1
        mw_ps_type_1          = task_model_mw("type_1", today, yesterday)
        
        # Malware Type2
        mw_ps_type_2          = task_model_mw("type_2", today, yesterday)

        # Malware Type3
        mw_ps_type_3          = task_model_mw("type_3", today, yesterday)
        
        # ip trend 
        # mw_ps_ip_trend        = task_model_mw("ip_trend", today, yesterday)
        # mw.get_save_trend(save_file_dir, mw_ps_ip_trend)

        # # save csv file
        # print(save_file_dir_type)
        save_file             = mw.get_save_df(save_file_dir_type, mw_ps_type_1, mw_ps_type_2, mw_ps_type_3 )

    def test_dos(self):
        from_date = '2020-01-02 04:00'
        to_date   = '2020-01-02 04:30'

        rt = task_model_dos(from_date, to_date)
        rt.to_csv(os.path.join(cst.PATH_DATA,"dos_ps_result_01.csv"))


if __name__ == "__main__":
    unittest.main()

