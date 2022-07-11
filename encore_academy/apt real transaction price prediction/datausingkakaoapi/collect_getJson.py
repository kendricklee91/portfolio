from .collection_funcs import *

jsondata = getJson()
finaldata = jsonProcessing(jsondata)

merge_test = pd.read_csv('merge_test_theater_subway_school_mart_kind_hospital.csv')
merge_test = merge_test.merge(finaldata, on = ['longitude', 'latitude'])

merge_test.to_csv('merge_test_theater_subway_school_mart_kind_hospital_government.csv', encoding = 'utf-8')