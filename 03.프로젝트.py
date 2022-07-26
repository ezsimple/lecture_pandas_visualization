#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl

# 한글화 작업
plt.figure(dpi=600) # 그래프를 선명하게
plt.rc('font', family = 'NanumGothic') # 시스템에 폰트설치후, 시스템 재시작
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결
plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# %%
file_name = 'data/202108_202108_연령별인구현황_월간.xlsx'

# %%
# 인구 피라미드
# https://jumin.mois.go.kr/ageStatMonth.do
df_m = pd.read_excel(file_name, skiprows=3, index_col='행정기관', usecols='B,E:Y')

# , 이외의 공백을 제거하고, 숫자로 변경
for idx in np.arange(df_m.shape[0]):
  df_m.iloc[idx] = df_m.iloc[idx].str.replace(',', '').astype(int)

df_m

# %%
df_w = pd.read_excel(file_name, skiprows=3, index_col='행정기관', usecols='B,AB:AV')
df_w.columns = df_w.columns.str.replace('.1', '')

# , 이외의 공백을 제거하고, 숫자로 변경
for idx in np.arange(df_w.shape[0]):
  df_w.iloc[idx] = df_w.iloc[idx].str.replace(',', '').astype(int)

df_w

# %%
plt.figure(figsize=(10,7))
plt.barh(df_m.columns, -df_m.iloc[0] // 1000, color='#006699', label='남자')
plt.barh(df_w.columns, df_w.iloc[0] // 1000, color='#ff6600', label='여자')
title = '2021년 8월 인구 피라미드'
plt.title(title)
plt.legend()
plt.savefig('data/'+title+'.png', dpi=200)

# %%
# 2. 출생아 수 및 합계출산율
# https://www.index.go.kr/potal/main/EachDtlPageDetail.do?idx_cd=1428

file_name = 'data/합계출산율_142801.xls'
df_b = pd.read_excel(file_name, skiprows=2, nrows=2, usecols='A:J', index_col=0)
df_b.index.values

# 인덱스에 있던 유니코드 공백을 제거
df_b.rename(index={'출생아\xa0수':'출생아수', '합계\xa0출산율':'합계출산율'}, inplace=True)
# df_b.loc['출생아수']

# %%
df_b = df_b.T

# %%
df_b

# %%
# X축을 공유하는 이중 라인 차트 생성
fig, ax1 = plt.subplots(figsize=(10,7))
ax1.plot(df_b.index, df_b['출생아수'])

ax2 = ax1.twinx() # x축을 공유하는 축을 생성
ax2.plot(df_b.index, df_b['합계출산율'], color='#ff6600')

# %%
# X축을 공유하는 이중 라인 차트 생성
fig, ax1 = plt.subplots(figsize=(10,5))
fig.suptitle('2021년 8월 출생아 수 및 합계출산율')
ax1.bar(df_b.index, df_b['출생아수'])
ax1.set_ylabel('출생아 수(천명)')
ax1.set_ylim(250, 700)
ax1.set_yticks([300, 400, 500, 600])
for idx, val in enumerate(df_b['출생아수']):
  ax1.text(idx, val+3, str(val), ha='center', va='bottom')

ax2 = ax1.twinx() # x축을 공유하는 축을 생성
ax2.plot(df_b.index, df_b['합계출산율'], color='#ff6600', marker='o', markersize=10, linewidth=3, markeredgecolor='#ffffff', markeredgewidth=2)
ax2.set_ylabel('합계출산율(가임여성 1명당 명)')
ax2.set_ylim(0, 1.5)
ax2.set_yticks([0, 1])
for idx, val in enumerate(df_b['합계출산율']):
  ax2.text(idx, val+0.05, str(val), ha='center', va='bottom')

plt.savefig('data/출생아수_합계출산율.png', dpi=200)


# %%
# Outro