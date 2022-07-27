#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# OpenAPI 주소
# https://www.data.go.kr/data/15084084/openapi.do


# %%
import requests

serviceKey = 'clzRha7FjiQHb9pLNqKTq1ieuSzvgbh+gIOGlrwUxQsVVk+fSJD5n5Ggu0YO3RDZEQowJ6eVgvZ65Hrw1C/+Fw=='
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
params ={'serviceKey' : serviceKey, 'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'JSON', 'base_date' : '20220727', 'base_time' : '0600', 'nx' : '55', 'ny' : '127' }

response = requests.get(url, params=params)
print(response.content.decode('utf-8'))