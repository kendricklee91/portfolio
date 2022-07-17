import pandas as pd
import json

class SynonymHandler():
    def __init__(self, synonym_path):
        self.path = synonym_path

    def synonym_handler(self, data_frame, col_name):
        mapper_dict = self.get_dict_json(col_name)

        if 'CD_INDUSTRY_C' == col_name:
            mapper_dict.pop('소')
        
        concat_df = pd.get_dummies(data_frame\
            .loc[(data_frame[col_name] != '') & ~(data_frame[col_name].isnull()), col_name]\
            .str\
            .split(',')\
            .apply(lambda x: self.transfer_code(mapper = mapper_dict, _list = x))\
            .explode(),
            prefix = col_name
        ).sum(level = 0)
        return concat_df
        
    def get_dict_json(self, col_name):
        file_name = self.path + col_name + '_dict.json'

        # json 파일의 위치경로 필요
        with open(file_name, 'r') as f:
            dict_file = json.load(f)
        return dict_file

    def transfer_code(self, mapper, _list):
        assert type(mapper) == dict, 'Error : mapper is not a dict type'
        assert type(_list) == list, print(_list) # 'Error : _list is not a list type'

        r_list = []
        for v in _list:
            r_list.append(mapper.get(v))
        return r_list
