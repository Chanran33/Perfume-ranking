# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 04:34:40 2022

@author: USER
"""
#데이터 수집
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd
from selenium import webdriver

result = []

rank = 0
for page in range(1,8):
    olive_url = 'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100050003&fltDispCatNo=&prdSort=01&pageIdx=%d&rowsPerPage=48&searchTypeSort=btn_thumb&plusButtonFlag=N&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0&trackingCd=Cat100000100050003_Small'%page
    print(olive_url)
    html = urllib.request.urlopen(olive_url)
    soupOlive = BeautifulSoup(html, 'html.parser')
    
    #wd = webdriver.Chrome('E:\빅데이터 기말발표\chromedriver.exe')
    
    for class_prdInfo in soupOlive.find_all('div',class_='prd_info'):
        #브랜드명
        tx_brand = class_prdInfo.find('span', class_='tx_brand')
        storeName = tx_brand.string
        #print(storeName)
        #제품명
        tx_name = class_prdInfo.find('p', class_='tx_name')
        productName = tx_name.string
        
        #제품순위
        rank += 1
        
        #제품용량
        find_capacity = productName.split()
        mls = [s for s in find_capacity if "ml" in s]
        if not mls:
            capacity = 0
        else:
            if '+' in  mls[0]:
                capacity = 0
            elif '(' in mls[0] :
                capacity = 0
            elif '/' in mls[0]:
                capacity = 0
            elif '*' in mls[0]:
                capacity = 0
            elif 'x' in mls[0]:
                capacity = 0
            else:
                capacity = mls[0]
            
        #제품가격
        tx_num = class_prdInfo.find('span', class_='tx_num')
        productPrice = tx_num.string
        #print(productPrice)
        result.append([productName]+[storeName]+[rank]+[capacity]+[productPrice])

df = pd.DataFrame(result, columns = ['productName','storeName','productRank','capacity','productPrice'])
print(df)
df.head(5)

#데이터 확인
df.to_csv('E:\빅데이터 기말발표\perfumeData.csv', index=False, encoding="utf-8-sig")
df.info()

#어떤 가게의 향수가 많은가
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')
print(df['storeName'].value_counts())
store = df['storeName'].value_counts().reset_index()
print(store)
plt.bar(store["index"], store["storeName"])
plt.xticks(rotation = 90)

#0ml로 파악할 수 없는 향수 drop 시키기
remove_data = df[df['capacity']==0].index
df.drop(remove_data, axis=0, inplace = True)

df.to_csv('E:\빅데이터 기말발표\perfumeData_refine.csv', index=False, encoding="utf-8-sig")

#용량별 가격 알아내기 위한 전처리
df['capacity']=df['capacity'].str.replace('ml','')
df['capacity']=df['capacity'].str.replace(',','')
df['productPrice']=df['productPrice'].str.replace(',','')
print(df['productPrice'])
df.to_csv('E:\빅데이터 기말발표\perfumeData_refine.csv', index=False, encoding="utf-8-sig")
df['capacity']= pd.to_numeric(df['capacity'])
df['productPrice']= pd.to_numeric(df['productPrice'])

#용량별 가격 알아내기
df['PriceByCapacity']=round((df['productPrice']/df['capacity']),2)
df.to_csv('E:\빅데이터 기말발표\perfumeData_refine.csv', index=False, encoding="utf-8-sig")

"""
순위 == 인기 라고 가정하고,
순위별 어떤 가격의 향수가 인기가 많은지를 
선형회귀 분석을 통해 알아보겠습니다.
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#최종적으로 정제된 데이터를 다시 불러와 dataframe에 넣어놓겠습니다.
perFume_df = pd.read_csv('E:\빅데이터 기말발표\perfumeData_refine.csv', encoding='utf-8', index_col=0, engine='python')

#가게 이름이 string이라 숫자로 붙여주겠습니다.
for i in range(60):
    perFume_df['storeName'] = perFume_df['storeName'].replace(store["index"][i],i)

#독립변수와 종속 변수 나누기
Y = perFume_df['productRank']
X = perFume_df.drop(['productRank'], axis=1, inplace = False)


#훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#선형회귀분석으로 모델 생성
lr = LinearRegression()

#모델 훈련
lr.fit(X_train, Y_train)

#선형회귀 분석 : 평가 데이터에 대한 예측 수행
Y_predict = lr.predict(X_test)

#분석 모델 시각화
"""
분석 평가 지표
- 회귀 분석 결과에 대한 평가 지표는 예측값과 실제값의 차이인 오류의 크기가 된다.
- 정확한 평가를 위해 오류의 절대값 평균이나 제곱의 평균, 제곱 평균의 제곱근 또는 분산 비율을 사용한다.
"""
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
print('Y 절편 값: ', lr.intercept_)
print('회귀 계수 값: ', np.round(lr.coef_, 1))

#분석모델 구축
coef = pd.Series(data = np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending = False)

#회귀식

#시각화 하기
import seaborn as sns
fig, axs = plt.subplots(figsize=(16, 16), ncols=2, nrows=2)
x_features = ['capacity', 'PriceByCapacity', 'productPrice', 'storeName']

for i, feature in enumerate(x_features):
      row = int(i/2)
      col = i%2
      sns.regplot(x=feature, y='productRank', data=perFume_df, ax=axs[row][col])