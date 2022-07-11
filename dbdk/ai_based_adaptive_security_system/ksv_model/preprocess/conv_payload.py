from operator import itemgetter
import urllib.parse as decode
from ksv_model.preprocess.rfc2616 import http_request_method, http_header_field, http_response_status_code, http_request_ver
import ksv_model.config.const as cst
# from ksv_model.preprocess.special_word import sw_sqli, sw_xss, sw_rce, sw_uaa, sw_fdo, sw_fup # for test

import pandas as pd
import numpy as np
import glob
import time
import csv
import re
import json
import os

"""
Data preprocessing
-----------
1. Data load & Split label and payload
2. Preprocess keywords and special word in payload
3. Change characters into ASCII code values
4. Add padding data
5. return payload_ps
"""
class KnownAttackPreprocess:

    def __init__(self):
        self.hrmk = list(http_request_method.keys())
        self.hrvk = list(http_request_ver.keys())
        self.hhfk = list(http_header_field.keys())
        self.hrsck = list(http_response_status_code.keys())

    def preprocess(self, df, attack_type, max_padd_size):
        buffer = []
        icnt = -1

        df = df

        if df.empty or df is None:
            print("Data None") # Error log level
            return False
        else:
            for _, row in df.iterrows():
                # 1. Split label and payload
                try:
                    label, payload = row[0], row[1]
                except IndexError:
                    print("split error") # Error log level
                    return False
                
                # 2. Preprocess keywords and special word in payload
                strTemp = self.preprocess_payload_str(attack_type, payload)
                
                icnt += 1
                buffer.append([])

                try:
                    buffer[icnt].extend([int(label)])
                except:
                    print("buffer error") # Error log level
                    return False
                
                # 3. Change characters into ASCII code values
                buffer[icnt] = self._asc_code_convert(strTemp, buffer[icnt])
            # 4. Add padding data
            payload_ps = self._padding_add(buffer, max_padd_size)

        return payload_ps

    """
    Preprocess one payload
    -----------
    1. Replace RFC2616 keywords with special character enclosed by string 'dbdk'
    2. Replace 'CCOMMAA' with ',' and lower payload
    3. From the payload, extract character list to process 
    4. Tokenize with extracted character list
    5. Tag pre-defined special words with string 'kdbddbdkddbbddkk'
        5-1. Remove 'dbdk' string from replacement of RFC2616 keywords
        5-2. Remove 'kdbddbdkddbbddkk' tag and leave only replacements characters for special words
    """
    def preprocess_payload_str(self, attack_type, payload):
        # 1. Replace RFC2616 keywords with special character enclosed by string 'dbdk'
        payload = self._replace_rfc2616_keyword(payload)

        # 2. Replace 'CCOMMAA' with ',' and lower payload
        payload = self._ccomma_lower(payload)

        # 3. From the payload, extract character list to process 
        lpFind = self._duplicate(payload)

        # 4. Tokenize with extracted character list
        payload_tokenize = self._tokenize(payload, lpFind)

        # 5. Tag pre-defined special words with string 'kdbddbdkddbbddkk'
        #   5-1. Remove 'dbdk' string from replacement of RFC2616 keywords
        #   5-2. Remove 'kdbddbdkddbbddkk' tag and leave only replacements characters for special words
        return self._specialWords(payload_tokenize, lpFind, attack_type)

    def asc_padding(self, payload_ps, max_padd_size, label=0):
        input_data = self._asc_code_convert(payload_ps, [])
        input_data.insert(0, int(label))
        return self._padding_add_one(input_data, max_padd_size)
    
    
    #### Internal use only
    def _get_data_df(self, load_file_dir, attack_type):
        if attack_type is None:
            return False

        file = load_file_dir.format(attack_type)
        # print(file) # for test
        df = pd.read_csv(file, encoding='utf-8')

        # use cols : label, payload
        df = pd.DataFrame(df[['label','payload']], columns=['label','payload'])

        # drop payload null
        df = df[~df['payload'].isnull()]
        # print('drop after data shape : {}'.format(df.shape)) # for test

        return df

    def _readFile(self, param):
        try:
            with open(param, encoding="utf-8", newline="\n") as f:
                reader = csv.reader(f)
                f_list = list(reader)
            return f_list
        except IOError:
            return None

    def _get_save_df(self, save_data, save_file_dir, attack_type):
        if attack_type is None:
            return False
        
        file = save_file_dir.format(attack_type)
        save_df = pd.DataFrame(save_data)
        if save_df.empty or save_df is None:
            print("Data None") # Error log level
            
            return False
        else:
            save_df.to_csv(file, index=False)
            
            return True

    def _saveFile(self, param, buffer):
        f_out = open(param, "w", newline="")
        saveFile = csv.writer(f_out)
        saveFile.writerows(buffer)
        f_out.close()
    
    def _ccomma_lower(self, payload):
        payload = payload.replace("CCOMMAA", ",")
        temppay = ""
        while (payload != temppay):
            temppay = payload
            payload = decode.unquote(payload)
        payload = payload.lower()

        return payload
    
    # Find all special characters from the payload and drop duplicates
    def _duplicate(self, payload):
        lpFind = list(set(re.findall(r'[\W_]', payload)))
        for ix in range(10):
            lpFind.append(ix)
        return lpFind
    
    # Tokenize with extracted character list
    def _tokenize(self, payload, lpFind):
        try:
            for token in lpFind:
                if token != " ":
                    payload = payload.replace(str(token), ' ' + str(token) + ' ').replace("  ", " ").replace('\r\n','').replace('\n','').replace('\r', '').replace('�', '')
            payload = " " + payload + " "
            payload = payload.strip()
            
            p = re.compile(r'dbdk\s\d\s\d\s\d\sdbdk')
            dbdk_list = p.findall(payload)
            
            for d in dbdk_list:
                payload = payload.replace(d, d.replace(' ', ''))
            return payload
        except:
            print("Tokenize Error") # Error log level
            return False

    def _replace_rfc2616_keyword(self, payload):
        # Replace http request keyword
        for s in self.hrmk:
            if payload.find(s) != -1:
                payload = payload.replace(s, ' dbdk'+str(http_request_method[s])+'dbdk ')

        # Replace http response keyword
        for reg in self.hrsck:
            exp = re.compile(reg)
            mat = exp.match(payload)
            if mat:
                payload = payload.replace(mat.group(), ' dbdk'+str(http_response_status_code[reg])+'dbdk ')

        # Replace http header keyword
        for s in self.hhfk:
            if payload.find(s) != -1:
                payload = payload.replace(s, ' dbdk'+str(http_header_field[s])+'dbdk ')

        # Replace http request version keyword: This should be the last process. 
        for s in self.hrvk:
            if payload.find(s) != -1:
                payload = payload.replace(s, ' dbdk'+str(http_request_ver[s])+'dbdk ')

        return payload

    def _specialWords(self, payload, lpFind, attack_type):
        # load sw json file
        sw_file = os.path.join(cst.PATH_CONFIG, 'special_words.json')
        with open(sw_file) as json_file:
            sw = json.load(json_file)
        
        # Set special words list according to the attack type
        if (attack_type == "sqli"):
            swords = sw['sw_sqli']
        elif (attack_type == "xss"):
            swords = sw['sw_xss']
        elif (attack_type == "rce"):
            swords = sw['sw_rce']
        elif (attack_type == "uaa"):
            swords = sw['sw_uaa']
        elif (attack_type == "fdo"):
            swords = sw['sw_fdo']
        elif (attack_type == "fup"):
            swords = sw['sw_fup']  
        # print(swords) # for test
                    
        # Set string length for each special word
        for i in range(len(swords)):
            iLenTemp = len(swords[i][0])
            swords[i][2] = iLenTemp

        lpCompare = []
        strTemp = ''
        px = re.compile(r'dbdk\d{3}dbdk')
        
        try:
            for word, change, _ in swords:
                payload = payload.replace(" " + word + " ", " kdbddbdkddbbddkk" + change + " ")
                lpCompare.append("kdbddbdkddbbddkk" + change)
                
            # leave only unique values
            lpCompare = list(set(lpCompare))
            resultPayload = payload.split(" ")  # Split by words
            
            for ChangeWord in resultPayload:
                if ChangeWord == ' ':
                    continue
                
                # Extract only the characters from RFC2616 replacement strings
                if px.match(ChangeWord):
                    strTemp = strTemp + ' ' + chr(int(ChangeWord[4:7]))
                    
                # Remove 'kdbddbdkddbbddkk' tag and leave only special word replacement characters and the original special characters
                elif ChangeWord in lpFind or ChangeWord in lpCompare:
                    strTemp = strTemp + " " + ChangeWord.replace("kdbddbdkddbbddkk", "")
            
            return strTemp
        except:
            print("Special Words Error") # Error log level
            return False
    
    # Change characters into ASCII code values
    def _asc_code_convert(self, strTemp, bufferCnt):
        try:
            for ch in strTemp:
                if ch != " ":
                    bufferCnt.append(ord(ch))
            return bufferCnt
        except:
            print("asc Error") # Error log level
            return False
    
    def _padding_add_one(self, input_data, max_padd_size):    
        iMax = max_padd_size

        payload_ps_len = len(input_data)
        padding0 = []
        padding0.append(input_data.pop(0))
        padding0.extend([0] * 2)

        if payload_ps_len < iMax:
            padding1 = [255] * (iMax - payload_ps_len)
            input_data = padding0 + input_data + padding1
        else:
            input_data = padding0 + input_data[0:iMax -1]

        input_data.extend([0] * 2)
        
        return input_data

    def _padding_add(self, buffer, padd_size):
        for i in range(len(buffer)):
            buffer[i] = self._padding_add_one(buffer[i], padd_size)

        return buffer
