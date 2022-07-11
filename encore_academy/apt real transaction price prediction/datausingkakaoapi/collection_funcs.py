from .collection_libs import *

def getCoordinates(csv):
    data = pd.read_csv(csv, encoding = 'utf8')
    coordinates = data.groupby(['latitude', 'longitude']).size()
    return coordinates

def getData(apikey, csv, query, distance):
    apikey = apikey

    coordinates = getCoordinates(csv)
    query = query
    distance = distance

    collect_data = []

    for c in coordinates:
        for d in distance:
            for q in query:
                result = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json?y=' + str(c[0]) + '&x=' + str(c[1]) + '&radius=' + str(d) +
                                    '&category_group_code=PO3&query=' + str(q), headers={"Authorization": apikey})
                data = result.json()
                dict = {"longitude": str(c[1]), "latitude": str(c[0]), "distance": str(d), "info": data}
                collect_data.append(dict)

    return collect_data

def makeJson(apikey, csv, query, distance):
    data = getData(apikey, csv, query, distance)

    with open('filename.json', 'w', encoding = 'utf-8-sig') as f: # 생성하고 싶은 filename으로 수정 가능
        json.dump(data, fp = f, ensure_ascii = False, indent = 3)

def getJson():
    with open('filename.json', 'r', encoding = 'utf-8-sig') as r:
        jsonData = json.load(fp = r)
    return jsonData

def jsonProcessing(jsondata):
    jsondata = jsondata

    jsondataList = []

    for i in range (len(jsondata)):
        v = {
            'longitude' : jsondata[i]['longitude'],
            'latitude' : jsondata[i]['latitude'],
            'distance' : jsondata[i]['distance'],
            'number' : jsondata[i]['info']['meta']['pageable_count']
        }
        jsondataList.append(v)

    jsondata_df = pd.DataFrame(jsondataList, columns = ['longitude', 'latitude', 'distance', 'number'])

    jsondata_df['???500'] = jsondata_df[jsondata_df['distance'] == '500']['number'] # ???에 변수명 입력
    jsondata_df['???1000'] = jsondata_df[jsondata_df['distance'] == '1000']['number']  # ???에 변수명 입력
    jsondata_df['???3000'] = jsondata_df[jsondata_df['distance'] == '3000']['number']  # ???에 변수명 입력
    jsondata_df['???5000'] = jsondata_df[jsondata_df['distance'] == '5000']['number']  # ???에 변수명 입력
    jsondata_df['???10000'] = jsondata_df[jsondata_df['distance'] == '10000']['number']  # ???에 변수명 입력

    jsondata_df = jsondata_df.fillna(0)
    jsondata_dist = jsondata_df.groupby(['longitude', 'latitude']).sum()

    jsondata_dist['???_500'] = jsondata_dist['???500'] # ???에 변수명 입력
    jsondata_dist['???_1000'] = jsondata_dist['???500']  # ???에 변수명 입력
    jsondata_dist['???_3000'] = jsondata_dist['???500']  # ???에 변수명 입력
    jsondata_dist['???_5000'] = jsondata_dist['???500']  # ???에 변수명 입력
    jsondata_dist['???_10000'] = jsondata_dist['???500']  # ???에 변수명 입력

    jsondata_dist_1 = jsondata_dist.drop(['number', '???500', '???1000', '???3000', '???5000', '???10000'], axis = 1) # ???에 변수명 입력
    jsondata_dist_final = jsondata_dist_1.reset_index()

    jsondata_dist_final.columns = ['longitude', 'latitude', '???_500', '???_1000', '???_3000', '???_5000', '???_10000']

    jsondata_dist_final['longitude'] = jsondata_dist_final['longitude'].astype(float)
    jsondata_dist_final['latitude'] = jsondata_dist_final['latitude'].astype(float)

    return jsondata_dist_final
