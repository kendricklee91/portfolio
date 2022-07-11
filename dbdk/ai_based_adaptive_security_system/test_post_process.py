from ksv_model.post_process import task_post_process_supervised, task_post_process_ip, task_post_process_blackip
import pandas as pd
import unittest


class TestPostProcess(unittest.TestCase):

    # Fixture
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # post process test of sqli injection model
    #@unittest.skip
    def test_sqli_post_process(self):
        model_type = 'sqli'
        date = '20191105'
        df = task_post_process_supervised(model_type=model_type, date=date, th_auto=True)
        columns = df.columns
        self.assertIn('check_policy', columns)

    # post process test of information gathering model
    #@unittest.skip
    def test_ig_post_process(self):
        pass

    # integrated post process test of IP based models (ig, dos, mw)
    #@unittest.skip
    def test_ip_post_process(self):
        df, _ = task_post_process_ip(date='20191206')
        columns = df.columns
        self.assertIn('mw_mds_i', columns)
    
    # post process test of black IP result
    #@unittest.skip
    def test_blackip_post_process(self):
        df = task_post_process_blackip(date='20191125')
        columns = df.columns
        self.assertIn('ip', columns)


if __name__ == "__main__":
    unittest.main()   
