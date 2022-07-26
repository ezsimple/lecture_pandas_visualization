#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# https://www.inflearn.com/course/%EB%82%98%EB%8F%84%EC%BD%94%EB%94%A9-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D-%EC%8B%9C%EA%B0%81%ED%99%94
from genericpath import exists
from locale import normalize
from timeit import timeit
from turtle import color
from matplotlib import legend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
import seaborn as sns
import matplotlib as mpl

# 한글화 작업
plt.figure(dpi=600) # 그래프를 선명하게
plt.rc('font', family = 'NanumGothic') # 시스템에 폰트설치후, 시스템 재시작
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결
plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# 판다스 최대 컬럼 지정 (컬럼에 ... 표시 방지)
pd.options.display.max_columns = 100
# retina 디스플레이가 지원되는 환경에서 시각화 폰트가 좀 더 선명해 보입니다.
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')


# %%
# pandas 데이터 분석 라이브러리
# Series : 1 차원 배열

temp = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
temp[-1:]

# Series 객체생성 (index 지정)
temp = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
temp['b':'f']


# %%
# Dictionary 데이터형
data = {
    '이름' : ['채치수', '정대만', '송태섭', '서태웅', '강백호', '변덕규', '황태산', '윤대협'],
    '학교' : ['북산고', '북산고', '북산고', '북산고', '북산고', '능남고', '능남고', '능남고'],
    '키' : [197, 184, 168, 187, 188, 202, 188, 190],
    '국어' : [90, 40, 80, 40, 15, 80, 55, 100],
    '영어' : [85, 35, 75, 60, 20, 100, 65, 85],
    '수학' : [100, 50, 70, 70, 10, 95, 45, 90],
    '과학' : [95, 55, 80, 75, 35, 85, 40, 95],
    '사회' : [85, 25, 75, 80, 10, 80, 35, 95],
    'SW특기' : ['Python', 'Java', 'Javascript', '', '', 'C', 'PYTHON', 'C#']
}
data

# %%
# DataFrame : 2 차원 배열(Series들의 집합)
# Dictionary 데이터형을 DataFrame 객체로 변환
df = pd.DataFrame(data)
df

# 데이터 접근
df[['이름', '키']]

# DataFrame에 인덱스 지정하기
df = pd.DataFrame(data, index=['1번', '2번', '3번', '4번', '5번', '6번', '7번', '8번'])
df

df.index.name = '지원번호'

# %%
# DataFrame에 컬럼 지정하기
df = pd.DataFrame(data, columns=['이름', '키', 'SW특기'])
df

# Index : 데이터에 접근할 수 있는 값
df[-1::-1]
df.index.name = '지원번호'
df

# %%
# index 초기화 ('지원번호)
df.reset_index(drop=True, inplace=True) # inplace : 실제 변경을 반영
df

#%%
# Index 설정 : 지정한 column으로 Index 설정
df.set_index('이름', inplace=True) # 이름으로 인덱스를 설정
df

# %%
# index를 기준으로 오름차순, 내림차순 정렬
df.sort_index(ascending=False, inplace=True)
df

# %%
df.sort_index(ascending=True, inplace=True)
df

# %%
# csv 저장하기
df
df.to_csv('data/score.csv')

# 텍스트파일로 저장
df.to_csv('data/score.txt', sep='\t')

# excel 저장하기 : openpyxl 라이브러리 사용
df.to_excel('data/score.xlsx')

# %%
df = pd.read_csv('data/score.csv', index_col='지원번호')
# df.set_index('지원번호', inplace=True)
df

# 1,2,3번째 rows를 제외하기
df = pd.read_csv('data/score.csv', skiprows=[2], nrows=4) # 2번째 row 제거하고, 4개의 rows를 가져오기
df

# %%
# 엑셀파일 읽어 오기
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df

# DataFrame 확인하기 (중요)
# 계산 가능한 데이터에 대한 컬럼별 갯수,평균,표준편차,최소,최대값을 출력
df.describe()

# %%
df.info()

# %%
df.tail()

# %%
df.values

# %%
df.shape

# %%
df.index

# %%
df.columns

# %%
# Series 확인
df['키'].describe()

# %%
df['키'].value_counts()

# %%
df['학교'].unique()

# %%
min_mean_max = (df['키'].min(), df['키'].mean().astype(int) ,df['키'].max())
min_mean_max

# 가장 큰 3명의 키를 출력
df['키'].nlargest(3)

# %%
# 데이터의 선택
# 엑셀파일 읽어 오기
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df

# 컬럼선택
df[['이름', '키']]

# 컬럼선택 - 정수 인덱스 사용
df[df.columns[-1]] # 맨끝의 컬럼을 선택

# %%
# 슬라이싱
df['영어'][0:5] # 0~4번째 컬럼을 선택

# %%
df[['이름', '키']][:3] # 처음 3명의 이름과 키를 선택

# %%
# 데이터의 선택
# 엑셀파일 읽어 오기
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df

# %%
# 이름을 이용하여 원하는 row에서 원하는 컬럼을 선택
df.loc['1번']

# %%
df.loc['1번', '국어']

#  %%
df.loc['1번', ['국어', '영어']]

# %%
df.loc[['1번', '2번'], ['국어', '영어']]

# %%
# 슬라이싱을 이용하여 원하는 row에서 원하는 컬럼을 선택
df.loc['1번':'3번', '국어':'수학']

# %%
# iloc : integer location
# 위치를 이용하여 원하는 row에서 원하는 컬럼을 선택

# 데이터의 선택
# 엑셀파일 읽어 오기
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df

# %%
df.iloc[0: 4] # 0번째 row를 선택


# %%
df.iloc[[0,1,2], [3,4,5,6,7]] # 0번째 row의 국어와 영어 컬럼을 선택

# %%
# iloc 슬라이싱
# df.iloc[[0,1,2], [3,4,5,6,7]] 와 동일
df.iloc[0:3, 3:8]

# %%
# 데이터 선택 (조건)
# 조건에 해당하는 데이타를 가져오기

# 엑셀파일 읽어 오기
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df

# 학생들의 키가 185이상 인지 여부를 체크
df[(df['키'] >= 187)] # 키가 187이상인 학생들을 선택

# %%
df[~(df['키'] >= 187)] # 키가 187이상이 아닌 학생들을 선택


# %%
df.loc[(df['키'] >= 187), '이름']

# %%
df.loc[df['키'] >= 187, ['이름', '수학']]

# %%
df.loc[(df['키'] >= 187) & (df['학교'] == '북산고'), ['이름', '영어', '수학']]

# %%
# str 함수를 이용하여 문자 조건 체크 예제
df[df['이름'].str.startswith('송')]

# %%
df[df['이름'].str.contains('태')]

# %%
df[~df['이름'].str.contains('태')]

# %%
lang = ['python', 'java']
filt = df['SW특기'].str.lower().isin(lang) # str.lower() => ignorecase
df[filt]

# %%
# 결측치(NaN) 처리 : na 옵션을 이용해서 처리 (중요)
# case 옵션 : 대소문자 구분 여부
filt = df['SW특기'].str.contains('java', case=False ,na=False)
df[filt]


# %%
# 결측치
# fillna
df.fillna('모름', inplace=True)
df

# %%
# dropna
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df.dropna(inplace=True)
df

# %%
# axis : 0 : 행, 1 : 열 OR 'index' : 행, 'columns' : 열
# how : 'any' : row에 하나라도 결측치가 있을 경우, 'all' : row의 모든값이 결측치일 경우
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df['학교'] = np.nan # 컬럼 전체를 NaN으로 채우기
df.dropna(axis='columns', how='all') # 학교 컬럼이 모두 NaN이므로, 컬럼 제거
df

# %%
# 데이터 정렬
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df = df.sort_values(by='키', ascending=False) # 키 컬럼을 기준으로 내림차순 정렬
df

# %%
df.sort_values(by=['영어', '수학'], ascending=False, inplace=True)
df

# 영어 컬럼을 기준으로 내림차순 정렬, 수학 컬럼을 기준으로 오름차순 정렬
df.sort_values(by=['영어', '수학'], ascending=[False, True], inplace=True)
df

# %%
df.sort_index(ascending=False) # 인덱스를 기준으로 내림차순 정렬

# %%
# 데이터 수정
# 컬럼 수정
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df['학교'].replace({'북산고': '상북고'}, inplace=True)
df

# %%
# 소문자 처리 및 결측지 처리
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df['SW특기'] = df['SW특기'].str.lower()
df['SW특기'] = df['SW특기'].fillna('모름')
df

# %%
# 컬럼 추가
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df['총합'] = df['국어'] + df['영어'] + df['수학'] + df['과학'] + df['사회']
df['평균'] = df['총합'] / 5
df

# 결과 컬럼 추가
df.loc[df['평균'] >= 90, '결과'] = 'Pass'
df['결과'] = df['결과'].fillna('Fail')
df

# %%
# 컬럼 삭제
df.drop(['총합'], axis='columns')
df

df.drop(index=['1번'])

# %%
filt = df['수학'] < 80
df[filt].index
df.drop(index=df[filt].index)
df

# %%
# row 추가
sum =94+ 89+ 91+ 90+ 87
avg = sum / 5
print(avg)
df.loc['9번'] = ['이정환', '해남고', 184, 94, 89, 91, 90, 87, 'Kotlin', sum, avg, 'Fail']
df.loc[df['평균'] >= 90, '결과'] = 'Pass'
df['결과'] = df['결과'].fillna('Fail')
df

# cell 수정
df.loc['4번', 'SW특기'] = 'python'
df

df.loc['5번', ['학교', 'SW특기']] = ['멍청고', 'C++']
df

# %%
# 컬럼 순서 변경
cols = list(df.columns)
cols

# 결과 컬럼을 맨 앞으로 이동
# 리스트 연산에 주의
df = df[[cols[-1]] + cols[0: -1]]

# %%
# 컬럼의 이름을 변경
# 컬럼명 존재 여부 체크 (set(df).issuperset(set['a', 'b', 'c']))
if set(['결과', '이름', '학교']).issubset(df.columns):
    df = df[['결과', '이름', '학교']]
    df.columns = ['Result', 'Name', 'School']
    df.index.name = 'No'
    df

# %%
# 함수 적용

df = pd.read_excel('data/score.xlsx', index_col='지원번호')

df['키'] = df['키'].apply(lambda x: str(x) + ' cm')
df

# %%
def addCM(x):
    v = str(x)
    if not v.endswith('cm'):
        return v + ' cm'
    return v

def capitalize(x):
    if not pd.notna(x):
        return x
    return x.capitalize()

df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df['키'] = df['키'].apply(addCM)
df['SW특기'] = df['SW특기'].apply(capitalize)
df


# %%
# 그룹화
df = pd.read_excel('data/score.xlsx', index_col='지원번호')
df.groupby('학교').get_group('북산고')
df.groupby('학교').mean()
df.groupby('학교').size()
df.groupby('학교').size()['능남고'] # 학교로 그룹화하고 능남고로 해당하는 데이터의 개수
df.groupby('학교')['키'].mean()
df.groupby('학교')[['국어', '영어', '수학']].mean()

df['학년'] = [3, 3, 2, 1, 1, 3, 2, 2] # 학년컬럼 추가
df.groupby(['학교', '학년']).mean()
df.groupby(['학년']).mean()
df.groupby(['학년']).mean().sort_values(by='키', ascending=False)

df.groupby(['학교', '학년']).sum() # 학교, 학년으로 분류하고 합계를 구함

df.groupby('학교')[['이름', 'SW특기']].count() # NaN은 카운터로 집계 되지 않습니다.
school = df.groupby('학교')
school[['학년']].value_counts() # 학교별 학년의 개수를 카운트함
school[['학년']].value_counts().loc['북산고'] # 북산고의 학년의 개고

school[['학년']].value_counts(normalize=True).loc['북산고'] # 북산고의 학년의 비율 (normalize=True)


# %%
# 퀴즈
data = {
    '영화' : ['명량', '극한직업', '신과함께-죄와 벌', '국제시장', '괴물', '도둑들', '7번방의 선물', '암살'],
    '개봉 연도' : [2014, 2019, 2017, 2014, 2006, 2012, 2013, 2015],
    '관객 수' : [1761, 1626, 1441, 1426, 1301, 1298, 1281, 1270], # (단위 : 만 명)
    '평점' : [8.88, 9.20, 8.73, 9.16, 8.62, 7.64, 8.83, 9.10]
}
qdf = pd.DataFrame(data)
qdf.to_csv('data/quiz.csv', index=False)


# %%
#1. 전체 데이터중 영화 정보만 출력하시오.
qdf[['영화']]


# %%
#2. 전체 데이터중에서 '영화', '평점' 정보를 출력하시오.
qdf[['영화', '평점']]


# %%
# 3. 2015년 이후에 개봉한 영화중 '영화' '개봉 연도'정보를 출력하시오.
qdf[qdf['개봉 연도'] > 2015]

# %%
# 4. 주어진 계산식을 참고하여, '추천점수' column을 추가 하시오
# 추천점수 = (관객수 * 평점) // 100
qdf['추천점수'] = qdf['관객 수'] * qdf['평점'] // 100
qdf


# %%
# 5. 전체 데이터를 '개봉연도' 기준 내림차순으로 정렬하시오
qdf.sort_values(by='개봉 연도', ascending=False)