---
layout: single
title: "네이버 금융 시계열형식의 유가데이터 크롤링"
date: 2020-05-16
comments: true
categories: 
    [Time-Series]
tags:
    [네이버금융, Web Crawling, Oil Price, read_html]
toc: false
publish: true
classes: wide
comments: true
---

웹사이트 크롤링에 대표적으로 사용하는 파이썬의 BeautifulSoup 과 Selenium 패키지 대신, Pandas 패키지에 연계된 read_html 함수를 이용해서 네이버 금융 웹사이트로 부터 유가정보를 가져오는 코드입니다. 

김나맥님의 원본코드 [링크](https://dacon.io/competitions/official/235606/codeshare/1037?page=1&dtype=recent&ptype=pub)


```python
#패키지 로드 및 정리를 위한 폴더 생성
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

if not any([s=='result' for s in os.listdir('.')]):
    os.mkdir('result')
if not any([s=='image' for s in os.listdir('.')]):
    os.mkdir('image')
```


```python
#함수에 입력하는 'code'변수의 예시입니다: OIL_CL = WTI(서부 텍사스 유), OIL_DU = 두바이유, OIL_LO = 경유

def get_oil_price(code):
    
    delay = 0.01
    page = 1
    result = []
    start_time = datetime.now()
    
    print('[{}] 데이터 수집 시작. (code: {})'.format(start_time.strftime('%Y/%m/%d %H:%M:%S'), code))
    while(True):
        url = 'https://finance.naver.com/marketindex/worldDailyQuote.nhn?marketindexCd={}&fdtc=2&page={}'.format(code, page)
        data = pd.read_html(url)[0].dropna()
        if page !=1:
            try:
                if data.iloc[-1,0] == result[-1].iloc[-1,0]:
                    break
            except:
                break
        result.append(data)
        page += 1
        time.sleep(delay)
    
    oil_price = pd.concat(result).reset_index(drop=True) #결과물을 데이터 프레임 형식으로 변환합니다
    oil_price.columns = ['날짜', '종가', '전일대비', '등락율']
    oil_price['날짜'] = oil_price['날짜'].apply(lambda x: datetime.strptime(x, '%Y.%m.%d'))
    oil_price = oil_price[['날짜', '종가']]
    oil_price.insert(0, '코드', code) #유가 종류에 대한 컬럼 생성
    
    end_time = datetime.now()
    print('[{}] 데이터 수집 종료. (code: {}, 수집시간: {}초, 데이터수: {:,}개)'.format(end_time.strftime('%Y/%m/%d %H:%M:%S'), code, (end_time-start_time).seconds, len(oil_price)))
    return oil_price
```


```python
oil_price_du = get_oil_price('OIL_DU')
oil_price_wti = get_oil_price('OIL_CL')
oil_price_brent = get_oil_price('OIL_BRT')
```

    [2020/05/16 10:51:35] 데이터 수집을 시작합니다. (code: OIL_DU)
    [2020/05/16 10:52:17] 데이터 수집을 종료합니다. (code: OIL_DU, 수집시간: 41초, 데이터수: 3,489개)
    [2020/05/16 10:52:17] 데이터 수집을 시작합니다. (code: OIL_CL)
    [2020/05/16 10:52:58] 데이터 수집을 종료합니다. (code: OIL_CL, 수집시간: 40초, 데이터수: 3,333개)
    [2020/05/16 10:52:58] 데이터 수집을 시작합니다. (code: OIL_BRT)
    [2020/05/16 10:53:37] 데이터 수집을 종료합니다. (code: OIL_BRT, 수집시간: 39초, 데이터수: 3,371개)



```python
oil_price_du
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>코드</th>
      <th>날짜</th>
      <th>종가</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>OIL_DU</td>
      <td>2020-05-15</td>
      <td>29.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OIL_DU</td>
      <td>2020-05-14</td>
      <td>29.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OIL_DU</td>
      <td>2020-05-13</td>
      <td>26.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OIL_DU</td>
      <td>2020-05-12</td>
      <td>26.81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OIL_DU</td>
      <td>2020-05-11</td>
      <td>27.27</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3484</th>
      <td>OIL_DU</td>
      <td>2006-04-24</td>
      <td>64.01</td>
    </tr>
    <tr>
      <th>3485</th>
      <td>OIL_DU</td>
      <td>2006-04-21</td>
      <td>63.70</td>
    </tr>
    <tr>
      <th>3486</th>
      <td>OIL_DU</td>
      <td>2006-04-20</td>
      <td>64.40</td>
    </tr>
    <tr>
      <th>3487</th>
      <td>OIL_DU</td>
      <td>2006-04-19</td>
      <td>65.10</td>
    </tr>
    <tr>
      <th>3488</th>
      <td>OIL_DU</td>
      <td>2006-04-18</td>
      <td>65.95</td>
    </tr>
  </tbody>
</table>
<p>3489 rows × 3 columns</p>
</div>




```python
#시각화단계

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10.5, 5))
plt.plot(oil_price_du['날짜'], oil_price_du['종가'], label='Dubai')
plt.plot(oil_price_wti['날짜'], oil_price_wti['종가'], label='WTI')
plt.plot(oil_price_brent['날짜'], oil_price_brent['종가'], label='Brent')

plt.title('Oil Price Trend')
plt.xlabel('Date')
plt.ylabel('Price(USD/Barrel)')
plt.legend()
plt.tight_layout()
plt.savefig('image/Oil Price Trend.jpg') #이미지 폴더내 파일 생성
plt.show()
```


![png](https://github.com/ohikendoit/ohikendoit.github.io/blob/8d6f7ab6207cd5000fa0dd20134e87e0a1edfef6/assets/image/Oil%20Price%20Trend.jpg?raw=true)

