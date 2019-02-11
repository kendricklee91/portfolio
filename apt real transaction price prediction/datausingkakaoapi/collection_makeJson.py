from .collection_funcs import *

apikey = '' # kakao developer api key

csv = 'preprocessing.csv'
query = [] # string 형태로 query 작성, ex) ['주민센터', '구청']
distance = [] # 위도 경도 기준 측정할 거리 값 작성, ex) [500, 1000, 3000, 5000, 10000]

if __name__ == "__main__":
    makeJson(apikey, csv, query, distance)