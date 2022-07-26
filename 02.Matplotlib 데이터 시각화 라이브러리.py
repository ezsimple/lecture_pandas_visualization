#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 다양한 형태의 그래프를 통해서 데이터를 시각화를 할 수 있는 라이브러리

# %%
from cProfile import label
from turtle import title, width
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from sklearn.covariance import shrunk_covariance

# 한글화 작업
plt.figure(dpi=600) # 그래프를 선명하게
plt.rc('font', family = 'NanumGothic') # 시스템에 폰트설치후, 시스템 재시작
plt.rc('font', size = 16) # 글자 크기 설정
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결
plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# %%
# 그래프 기본
x=[1,2,3]
y=[2,4,8]
plt.plot(x,y)
plt.title('라인 그래프')

# %%
# 축
plt.title('꺾은선 그래프')
plt.xlabel('x축', color='r', loc='right') # left, center, right
plt.ylabel('y축', color='#0000ff', loc='top') # top, center, bottom
plt.plot(x,y)
plt.xticks([1,2,3])
plt.yticks([3,6,9])

# %%
# 범례
plt.plot(x, y, label='범례')
plt.legend(loc='lower right') # lower, upper, center
plt.legend(loc=(0.6, 0.5)) # 좌표를 설정해서 범례를 위치시킬 수 있음

# %%
# 스타일
# plt.plot(x,y,linewidth=1, marker='o', linestyle='None')

# 마커 스타일
# https://matplotlib.org/stable/api/markers_api.html
plt.plot(x,y,linewidth=1, marker='H', markersize=20, markeredgecolor='r', markerfacecolor='yellow')

# %%
# 라인 스타일
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
plt.plot(x, y, linestyle='-.', linewidth=1)

# %%
# 컬러 스타일
# https://matplotlib.org/stable/tutorials/colors/colors.html
# plt.plot(x, y, 'ro--', linewidth=1) # ro-- : color marker linestyle
# plt.plot(x, y, 'bv:' , linewidth=1) # bv: : color marker linestyle
plt.plot(x, y, 'go', linewidth=1) # go : color marker linestyle(None)

# %%
# 축약어
# mfc : marker face color, mew : marker edge width, ls : linestyle
# https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html
plt.plot(x,y, marker='o', mfc='r', ms=10, mec='b', mew=2, ls=":", alpha=0.5)

# %%
# 그래프 크기
plt.figure(figsize=(10,5), dpi=200) # dpi : dot per inch 확대율
plt.plot(x,y)

# 배경색
plt.figure(facecolor='#a1c1ff')
plt.plot(x,y)

# %%
# 파일 저장
plt.plot(x,y)
plt.savefig('data/test.png', dpi=150)


# %%
# 텍스트
plt.plot(x, y, marker='o')
for idx, txt in enumerate(y):
  plt.text(x[idx], y[idx] + 0.3, txt, ha='center', va='bottom')

# %%
# 여러 데이터
days = [1, 2, 3]
az = [2, 4, 8]
pfizar = [3, 6, 9]
moderna = [4, 8, 12]
plt.plot(days, az, label='AZ')
plt.plot(days, pfizar, label='PFIZAR')
plt.plot(days, moderna, label='MODERNA')
plt.legend(loc='upper left')
plt.xticks(days)

# %%
# 막대 그래프(기본)
labels = ['강백호' , '서태웅', '정대만']
values = [190, 187, 184]
colors = ['r', 'g', 'b']
plt.bar(labels, values, color=colors, alpha=0.5)

# %%
# 크기가 비슷해서 판단이 어려울 경우
plt.bar(labels, values, width=0.3)
plt.ylim(175, 195) # y축의 범위를 제한해서 그래프를 도드라지게 보이게 하는 기법
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# %%
# 레이블을 다른 값으로 대체 하기
ticks = ['1번학생', '2번학생', '3번학생']
plt.bar(labels, values)
plt.xticks(labels, ticks)
plt.show()

# %%
# 막대 그래프(심화)
bar = plt.barh(labels, values, color=colors, alpha=0.5)
plt.xlim(175, 195) # y축의 범위를 제한해서 그래프를 도드라지게 보이게 하는 기법

# 패턴 채우기
# https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html
bar[0].set_hatch('/')
bar[1].set_hatch('*')
bar[2].set_hatch('--')

# %%
# 텍스트 채우기
bar = plt.bar(labels, values, color=colors, alpha=0.5)
plt.ylim(175, 195) # y축의 범위를 제한해서 그래프를 도드라지게 보이게 하는 기법
plt.yticks(values)

for idx, rect in enumerate(bar):
  plt.text(idx, rect.get_height(), values[idx], ha='center', va='bottom')

# %%
# DataFrame 활용
df = pd.read_csv('data/score.csv')
df

# %%
plt.plot(df['지원번호'], df['국어'], label='국어', linewidth=1)
plt.plot(df['지원번호'], df['영어'], label='영어', linewidth=1)
plt.plot(df['지원번호'], df['수학'], label='수학', linewidth=1)
plt.plot(df['지원번호'], df['과학'], label='과학', linewidth=1)
plt.plot(df['지원번호'], df['사회'], label='사회', linewidth=1)
plt.legend(loc='upper left')
plt.grid(axis='x', linestyle="--")

# %%
# 누적 막대 그래프
plt.bar(df['이름'], df['국어'], label='국어', color='r', alpha=0.5)
plt.bar(df['이름'], df['영어'], bottom=df['국어'], label='영어', color='g', alpha=0.5)
plt.bar(df['이름'], df['수학'], bottom=df['국어'] + df['영어'],  label='수학', color='b', alpha=0.5)
plt.legend(loc='best')
plt.xticks(rotation=45)

# %%
# 다중 막대 그래프
N = df.shape[0] # 행의 개수
index = np.arange(N)

w = 0.25
plt.figure(figsize=(10, 5))
plt.bar(index - w, df['국어'], width=w, label='국어', color='r', alpha=0.5)
plt.bar(index, df['영어'], width=w, label='영어', color='g', alpha=0.5)
plt.bar(index + w, df['수학'], width=w, label='수학', color='b', alpha=0.5)
plt.legend(loc='best', ncol=3)
plt.xticks(index, df['이름'], rotation=45)
plt.show()

# %%
# 원 그래프 (기본)
values = [30, 25, 20, 13, 10, 2] # 100% 비율로 사용하게 됩니다.
lables = ['Python', 'Java', 'Javascript', 'C#', 'C/C++', 'ETC']
plt.pie(values, labels=lables, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.show()

# %%
# 간격 띄워서 그리기
explode = [0.05] * len(values)
plt.pie(values, labels=lables, autopct='%1.1f%%', explode=explode)
plt.legend(title='언어별 선호도', loc=(1.2, 0.3))
plt.show()

# %%
# 원 그래프 (심화)
values = [30, 25, 20, 13, 10, 2] # 100% 비율로 사용하게 됩니다.
lables = ['Python', 'Java', 'Javascript', 'C#', 'C/C++', 'ETC']
# colors = ['r', 'g', 'b', 'c', 'm', 'y'] # 눈아파요
colors = ['#ffadad', '#ffd6a5', '#fdffb6', '#caffbf', '#9bf6ff', '#a0c4ff']
explode = [0.02] * len(values)
plt.pie(values, labels=lables, colors=colors,autopct='%1.1f%%', explode=explode)
plt.legend(title='언어별 선호도', loc=(1.2, 0.3))
plt.show()

# %%
# wedgeprops 를 이용해서 도넛 차트 그리기
wedgeprops = {'width':0.4, 'edgecolor':'w', 'linewidth':1}
plt.pie(values, labels=lables, colors=colors,autopct='%1.1f%%', wedgeprops=wedgeprops)
plt.legend(title='언어별 선호도', loc=(1.2, 0.3))
plt.show()

# %%
# 10% 이상만 출력하는 커스텀 함수
def custom_autopct(pct):
  # return ('%1.1f%%' % pct) if pct > 10 else ''
  return '{:.1f}%'.format(pct) if pct > 10 else ''

wedgeprops = {'width':0.5, 'edgecolor':'w', 'linewidth':1}
plt.pie(values, labels=lables, colors=colors
  , autopct=custom_autopct
  , wedgeprops=wedgeprops
  , pctdistance=0.75
  )
plt.legend(title='언어별 선호도', loc=(1.2, 0.3))
plt.show()

# %%
# DataFrame 활용
df = pd.read_excel('data/score.xlsx')
df

grp = df.groupby('학교')
grp.size()['북산고']
grp.size().plot(kind='pie', figsize=(10, 5), autopct='%1.1f%%', title='소속 학교')

# %%
values = [grp.size()['북산고'], grp.size()['능남고']]
lables = ['북산고', '능남고']

plt.pie(values, labels=lables, autopct='%1.1f%%')
plt.title('소속 학교')

# %%
# 산점도
df = pd.read_excel('data/score.xlsx')
df['학년'] = [3,3,2,1,1,3,2,2]
df

# %%
plt.figure(figsize=(10, 10))
sizes = df['학년'] * 500
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.scatter(df['수학'], df['영어'], s=sizes, c=df['학년'], cmap='viridis', alpha=0.3)
plt.xlabel('수학')
plt.ylabel('영어')
plt.colorbar(ticks=[1,2,3], label='학년', shrink=0.5, orientation='horizontal')

# %%
# 여러 그래프
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # 2 x 2 그래프
fig.suptitle('여러 그래프')

g0 = axs[0, 0]
g1 = axs[0, 1]
g2 = axs[1, 0]
g3 = axs[1, 1]

# bar chart
g0.bar(df['이름'], df['수학'], label='수학점수')
g0.set_title('수학')
g0.set_xlabel('이름')
g0.set_ylabel('점수')
g0.set_xticklabels(df['이름'], rotation=45)
g0.set_facecolor('lightyellow')
g0.grid(linestyle='--', alpha=0.5)
g0.legend()

# line chart
g1.plot(df['이름'], df['영어'], label='영어')
g1.plot(df['이름'], df['수학'], label='수학')
g1.set_xticklabels(df['이름'], rotation=45)
g1.legend()

# barh chart
g2.barh(df['이름'], df['키'], label='키')
g2.set_xlim(150, 200)

# pie chart
g3.pie(df['수학'], labels=df['이름'], autopct='%1.1f%%', wedgeprops={'width':0.7, 'edgecolor':'w', 'linewidth':1})

# %%
# 퀴즈
data = {
    '영화' : ['명량', '극한직업', '신과함께-죄와 벌', '국제시장', '괴물', '도둑들', '7번방의 선물', '암살'],
    '개봉 연도' : [2014, 2019, 2017, 2014, 2006, 2012, 2013, 2015],
    '관객 수' : [1761, 1626, 1441, 1426, 1301, 1298, 1281, 1270], # (단위 : 만 명)
    '평점' : [8.88, 9.20, 8.73, 9.16, 8.62, 7.64, 8.83, 9.10]
}
qdf = pd.DataFrame(data)
qdf.to_excel('data/quiz_graph.xlsx')
qdf


# %%
# 1. 영화 데이터를 활용해서 x측은 영화, y축은 평점인 막대 그래프를 그리자
plt.bar(qdf['영화'], qdf['평점'])
plt.show()

# %%
# 2. 앞에서 만든 그래프에 제시된 세부사항을 적용하시오
# 제목 : 국내 Top 8 영화 평점 정보
# x축 label : 영화(90도 회전)
# y축 label : 평점
plt.bar(qdf['영화'], qdf['평점'], width=0.5)
plt.title('국내 Top 8 영화 평점 정보')
plt.xticks(rotation=90)
plt.xlabel('영화')
plt.ylabel('평점', rotation=0)
plt.ylim(7, 10)

# %%
# 3. 개봉 연동별 평점 변화 추이를 꺾은선 그래프로 그리시오
# 연도별 평균 데이터를 구하는 코드는 다음과 같습니다.
# df_grp = df.groupby('개봉 연도').mean()
# df_grp
df_grp = qdf.groupby('개봉 연도').mean()
df_grp

plt.plot(df_grp.index, df_grp['평점'])

# %%
# 4. 앞에서 만든 그래프에 제시된 세부사항을 적용하시오
# marker: 'o'
# x 축 눈금 : 5년마다 표시
# y 축 눈금 : 최소7, 최대 10
plt.figure(figsize=(10, 5))
plt.plot(df_grp.index, df_grp['평점'], marker='o')
xlables = np.arange(2005, 2021, 5)
plt.xticks(xlables)
plt.ylim(7, 10)
xlables

# %%
# 5. 평점이 9점 이상인 영화의 비율을 확인할 수 있는 원 그래프를 제시된 세부사항을
# 적용하여 그리시오.
# label: 9점 이상 / 9점 미만
# 퍼센트 : 소숫점 첫째자리까지 표시
# 범례 : 그래프 우측에 표시
# x = qdf[qdf['평점'] >= 9]['영화'].values
# y = qdf[qdf['평점'] >= 9]['평점'].values

filt = qdf['평점'] >= 9
# x = qdf[filt]['영화'].count()
# y = qdf[~filt]['영화'].count()
values = [len(df[filt]), len(df[~filt])]
plt.pie(values, labels=['9점 이상', '9점 미만'], autopct='%1.1f%%')
plt.legend(loc=(1.0, 0.4))
