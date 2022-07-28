#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
from encodings.utf_8 import encode
from turtle import position
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.transforms as transforms

# 한글화 작업
plt.figure(dpi=600) # 그래프를 선명하게
plt.rc('font', family = 'NanumGothic') # 시스템에 폰트설치후, 시스템 재시작
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결
plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# 기상청 지역별 평균기온 및 강수량 시각화
from datetime import datetime, timedelta
import pandas as pd
import requests as r
import json

serviceKey='clzRha7FjiQHb9pLNqKTq1ieuSzvgbh+gIOGlrwUxQsVVk+fSJD5n5Ggu0YO3RDZEQowJ6eVgvZ65Hrw1C/+Fw=='
pageNo=1
numOfRows=31
startDt='20220601'
endDt='20220630'
stnIds=112 # 서산(129), 인천(112), 서울(108), 홍천(212)
# 당진 <-> 대전(125), 부여(113), 보령(83), 서산(31), 금산(135), 천안(51) km 이내

# 날짜 유효성 검사 및 Fix
today = datetime.today()
first_day = today.strftime('%Y%m01')
yesterday = today - timedelta(days=1)
yesterday = yesterday.strftime('%Y%m%d')

year = today.strftime('%Y')
month = today.strftime('%m')
day = today.strftime('%d')

# 당월 마지막 날짜
import calendar
first_day_of_month = '01'
last_day_of_month = calendar.monthrange(int(year), int(month))[1]

startDt = year + month + first_day_of_month
endDt = year + month + str(last_day_of_month)

if startDt > yesterday:
  startDt = first_day

if endDt > yesterday:
  endDt = yesterday

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
params ={'serviceKey' : serviceKey, 'pageNo' : pageNo, 'numOfRows' : numOfRows, 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'DAY', 'startDt' : startDt, 'endDt' : endDt, 'stnIds' : stnIds  }

try:
  res = r.get(url, params=params)
  s = res.content.decode('utf-8')
  dict = json.loads(s)
  result_msg = dict.get('response').get('header').get('resultMsg')
  if result_msg != 'NORMAL_SERVICE':
    raise Exception(result_msg)

  item = dict.get('response').get('body').get('items').get('item')
except Exception as e:
  print(e)

# minTa: 최저 기온 , maxTa: 최고 기온 , sumRn: 일강수량
df = pd.DataFrame(item)
df.set_index('tm', inplace=True) # tm: 일시
df.rename(columns={'avgTa':'평균 기온', 'minTa': '최저 기온', 'maxTa': '최고 기온', 'sumRn': '일강수량'}, inplace=True)
df.index.name = '일자'
df = df.replace(r'^\s*$', np.nan, regex=True) # 공백제거
df['일강수량'] = df['일강수량'].fillna(0.0) # 결측치 초기화

# 타입 변환
df['일강수량'] = df['일강수량'].astype(float)
df['평균 기온'] = df['평균 기온'].astype(float)
df['최고 기온'] = df['최고 기온'].astype(float)
df['최저 기온'] = df['최저 기온'].astype(float)

# 날짜 추출
dt = pd.to_datetime(df.index)
df['년도'] = dt.year
df['월'] = dt.month
df['일'] = dt.day

# 중간 10개의 온도 추출
# all_temps = df['평균 기온'].values.astype(float)
# max_temps = df.sort_values(by='평균 기온', ascending=False).head(10)['평균 기온'].to_list()
# min_temps = df.sort_values(by='평균 기온', ascending=True).head(10)['평균 기온'].to_list()

# temps = pd.Series(all_temps)
# for value in max_temps:
#     temps.drop(temps[temps == value].index, inplace=True)
# for value in min_temps:
#     temps.drop(temps[temps == value].index, inplace=True)

mean_temp = df['평균 기온'].mean()
max_temp = df['평균 기온'].max()
min_temp = df['평균 기온'].min()

# 중간값중 중복값 제거
# temps = temps.drop_duplicates()
# temp_max = temps.max()
# temp_min = temps.min()

# 지역코드 정의
df_city = pd.read_csv('data/city.csv', encoding='utf-8')
df_city.set_index('id', inplace=True)
city_name = df_city.loc[stnIds]['name']

# X축을 공유하는 이중 라인 차트 생성
fig, ax1 = plt.subplots(figsize=(10,6), sharex='col')
title=f"{city_name} 평균기온 및 일강수량 ({startDt} ~ {endDt})"
fig.suptitle(title)
ax1.plot(df.index, df['최고 기온'], color='#ff6600', marker='^', ls=':' ,markersize=10, linewidth=1, markeredgecolor='#ffffff', markeredgewidth=2, label='최고기온')
ax1.plot(df.index, df['평균 기온'], color='#ffc6c0', marker='o', ls='-' ,markersize=10, linewidth=3, markeredgecolor='#ffffff', markeredgewidth=2, label='평균기온')
ax1.plot(df.index, df['최저 기온'], color='#0066cc', marker='v', ls=':' ,markersize=10, linewidth=1, markeredgecolor='#ffffff', markeredgewidth=2, label='최저기온')

ax1.set_ylabel('최고/평균/최저 기온(°C)')
ax1.yaxis.set_label_coords(-0.07, 0.5)

ax1.set_xticklabels(df['일'], ha='center', rotation=0)
ax1.legend(loc=(1.1, 0.80))
for idx, val in enumerate(df['평균 기온']):
  if val >= max_temp: #
    ax1.text(idx, val, str(val), ha='center', va='bottom')

# 평균 온도를 수평 라인으로 표시
ax1.axhline(mean_temp, color="red", linestyle=":", linewidth=2, alpha=0.5)
trans = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
ax1.text(0,mean_temp, "{:.1f}".format(mean_temp), color="red", transform=trans, ha="right", va="center")

# 하단에 레이블 표시 (footer)
ax1.set_xlabel('과거 일자별 날씨정보 (공공 데이터 이용 자료)')

ax2 = ax1.twinx() # x축을 공유하는 축을 생성
p2 = ax2.bar(df.index, df['일강수량'], color='#0033ff', alpha=0.5, label='일강수량')
ax2.set_ylabel('일강수량(mm)')
ax2.legend(loc=(1.1, 0.74))
for idx, val in enumerate(df['일강수량']):
  if val >= 1.0: # 1mm 이상만 출력
    ax2.text(idx, val+0.05, str(val), ha='center', va='bottom')

# 그래프 저장
fig.savefig('data/'+title+'.png', dpi=300)
