from ksv_model.preprocess.network_analysis import NetworkAnalysis
import ksv_model.config.const as cst
import os
import pandas as pd
import pickle
import unittest


class TestNetworkAnalysis(unittest.TestCase):

    # Fixture
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # network analysis test
    #@unittest.skip
    def test_network_analysis(self):
        # load ip data from pickle file
        with open(os.path.join(cst.PATH_DATA, 'df_ip_nx_test.pkl'), 'rb') as f:
            df_ip = pickle.load(f)

        netx = NetworkAnalysis()
        print(netx.get_high_degree_centrality_ip(df_ip, 0.1))


if __name__ == "__main__":
    unittest.main()   
