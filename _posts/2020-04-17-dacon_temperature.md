---
layout: single
title: "데이콘 기상센서 활용 온도추정 대회 결과"
date: 2020-04-17
comments: true
categories: 
    [PROJECT]
tags:
    [Algorithm, LSTM, Competition]
toc: false
publish: true
classes: wide
comments: true
permalink: /dacon_temperature
---

지난 3월1일부터 4월 13일까지 약 한달이 조금 넘는 기간동안 데이콘 플랫폼에서 [온도 추정 경진대회]가 열렸습니다. 본 대회는 온도측정이 가능한 IOT 센서가 도시 곳곳에 설치되 10분단위로 온도를 측정한다는 가정하에, 이를 기상청에서 제공하는 공공데이터와 상관관계 모델을 만들어 관측 데이터만으로 해당 센서의 온도를 추정하는것이 목표었습니다.

연구교류모임인 AI프렌즈, 한국원자력연구원, 한국기계연구원이 주최하고, 연구개발특구진흥재단이 후원한 대회입니다. 최근 데이콘에서 열리는 데이터분석 대회의 테마와 카테고리가 점점 다양해지고 있어서 반갑습니다. 대회참가에서도 모델 생성으로 그 과정이 끝나는것이 아니라 리더보드 상위권에 진입하려면 최적화를 통해 점수를 최대한 끌어올려야하고 하루에 정답 제출횟수가 제한되어 꾸준한 실험정신이 필요한것같습니다.

```
Dacon.io AI프렌즈 시즌1 온도 추정 경진대회
케라스 LSTM 모델 활용을 통한 센서 온도 예측
- Sun_A, ohikendoit
```
대회 시작 후 일주일간은 결과물을 매일 하나씩 제출하다 베이스라인 코드를 뛰어넘는 점수를 만들지 못해 투두리스트 한켠으로만 두고있었습니다. 4월이 되서야 본격적으로 데이터분석 파트너 Sun_A 님과 함께 대회 참여를 시작했습니다. 예측 및 분류 테스크에 좋은 결과를 내는 모델이 아닌 시계열 데이터에 최적화된 LSTM 레이어를 통해 모델을 생성했고 파생변수들을 만들었습니다. 최종적으로  **992개의 참가팀 중 41등**이라는 성적으로 마무리할 수 있었습니다.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate,LSTM, Flatten
from keras.models import Model
```
학습과 테스트용 데이터로는 대전 지역에서 측정된 실내외 19곳의 온도데이터와 주변지역의 기상청 공공데이터를 비식별화해 제공 받았습니다. 30일간의 기성청 데이터 (X00~X39) 와 센서데이터 (Y00~Y17) 이 주어졌고 이후 3일간의 기상청 데이터(X00~X39) 및 센서데이터 (Y18) 가 있었습니다. 보다 정확한 Y18값을 예측하기위해서는 3일간의 데이터 보다는 주어진 전체 33일간의 데이터를 활용해야함으로 주어지지않은 Y18값을 위해 어떻게 전이학습을 적용하는지가 하나의 관권이었던거 같습니다.
    
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#고장난 센서 데이터 컬럼 삭제 
def same_min_max(df):
    return df.drop(df.columns[df.max() == df.min()], axis=1)

test = same_min_max(test)
train = same_min_max(train)


#주어지지않은 Y18값 유추를 위해 상관관계가 높은 컬럼 6개의 평균값으로 대체
train = train.drop(['id'], axis=1)
train['Y_combined']=train[['Y15','Y11','Y09','Y17','Y10','Y16']].mean(axis=1)
# train = train.drop(['id'], axis=1)

#Y18값이 존재하는 부분은 유지, 존재하지 않은 경우 평균값으로 추가
train['Y']=list(train[train['Y_combined'].notna()]['Y_combined'])+list(train[train['Y18'].notna()]['Y18'])
train = train.drop(['Y_combined'], axis=1)
```


```python
# 강수량 컬럼 실측치를 강수여부를 나타내는 이진법으로 전환하는 함수
def precipitation_bi(DF,x):
    DF[x+'_b'] = 0
    for j,i in enumerate(DF[x]):
        if i==0:
            DF[x+'_b'].iloc[j] = 0
        else:
            DF[x+'_b'].iloc[j] = 1
```

기상청 데이터로는 기온, 현지기압, 풍속, 일일 누적강수량, 해면기압, 일일 누적일사량, 습도, 풍향이 있었으며 EDA 과정을 통해 비가 오고안옴이 해당 날짜의 온도변화에 가장 큰 영향을 미친다는것을 확인 할 수 있었습니다. 따라서 해당 데이터에 대한 새로운 모델인풋을 생성하게되었고 이에 따라 모델 성능이 확연이 나아진것을 확인할 수 있었습니다.

```python
temperature_cols = ['X00','X07','X28','X31','X32'] #기온 컬럼
pressure_cols = ['X01', 'X06', 'X22', 'X27', 'X29'] #현지기압 컬럼
windspeed_cols = ['X02', 'X03', 'X18', 'X24', 'X26'] #풍속 컬럼
precipitation_cols = ['X04', 'X10', 'X21', 'X36', 'X39'] #일일 누적강수량 컬럼
sealevelpress_cols = ['X05', 'X08', 'X09', 'X23', 'X33'] #해면기압 컬럼
# sunshie_cols = ['X11','X14','X16','X19','X34'] #일일 누적일사량 컬럼
sunshie_cols = ['X11','X34'] #일일 누적일사량 중 데이터가 존재하는 컬럼
humidity_cols = ['X12','X20','X30','X37','X38'] #습도 컬럼
winddirection_cols = ['X13','X15','X17','X25','X35'] #풍향 컬럼
precip_b_cols = ['X04_b', 'X10_b', 'X21_b', 'X36_b', 'X39_b'] #강수여부 추가컬럼
```


```python
for x in precipitation_cols:
    precipitation_bi(train,x)
    precipitation_bi(test,x)
```


```python
#10분단위로 주어진 데이터에 적용할 시간 변수 생성
minute=pd.Series((train.index%144).astype(int))
hour=pd.Series((train.index%144/6).astype(int))

min_in_day = 24*6
hour_in_day = 24

minute_sin = np.sin(np.pi*minute/min_in_day)
minute_cos = np.cos(np.pi*minute/min_in_day)

hour_sin = np.sin(np.pi*hour/hour_in_day)
hour_cos = np.cos(np.pi*hour/hour_in_day)

#하루사이클을 담고있는 삼각함수
t1 = range(len(minute_sin[:144]))
plt.plot(t1, minute_sin[:144], 
         t1, minute_cos[:144], 'r-')
plt.title("Sin & Cos")
plt.show()
```


![png](/assets/image/output_7_0.png)



```python
time1 = np.array([i for i in minute_sin[:144]]*33)
time2 = np.array([i for i in minute_cos[:144]]*33)

time = np.array([(i,j) for i,j in zip(time1,time2)])
```


```python
temp = np.array(train[temperature_cols]).reshape(-1,1,5)
press = np.array(train[pressure_cols]).reshape(-1,1,5)
winds = np.array(train[windspeed_cols]).reshape(-1,1,5)
precip = np.array(train[precipitation_cols]).reshape(-1,1,5)
sea = np.array(train[sealevelpress_cols]).reshape(-1,1,5)
sun = np.array(train[sunshie_cols]).reshape(-1,1,2)
humid = np.array(train[humidity_cols]).reshape(-1,1,5)
windd = np.array(train[winddirection_cols]).reshape(-1,1,5)
precip_b = np.array(train[precip_b_cols]).reshape(-1,1,5)

trainY = train['Y']
```


```python
#LSTM 모델 생성
tempInput = Input(batch_shape=(None, 1, 5))
tempLstm = LSTM(100)(tempInput)
tempDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(tempLstm)
tempDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(tempDense_1)


pressInput = Input(batch_shape=(None, 1, 5))
pressLstm = LSTM(100)(pressInput)
pressDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(pressLstm)
pressDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(pressDense_1)

windsInput = Input(batch_shape=(None, 1, 5))
windsLstm = LSTM(100)(windsInput)
windsDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(windsLstm)
windsDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(windsDense_1)

precipInput = Input(batch_shape=(None, 1, 5))
precipLstm = LSTM(100)(precipInput)
precipDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(precipLstm)
precipDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(precipDense_1)

seaInput = Input(batch_shape=(None, 1, 5))
seaLstm = LSTM(100)(seaInput)
seaDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(seaLstm)
seaDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(seaDense_1)

sunInput = Input(batch_shape=(None, 1, 2))
sunLstm = LSTM(100)(sunInput)
sunDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(sunLstm)
sunDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(sunDense_1)

humidInput = Input(batch_shape=(None, 1, 5))
humidLstm = LSTM(100)(humidInput)
humidDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(humidLstm)
humidDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(humidDense_1)

winddInput = Input(batch_shape=(None, 1, 5))
winddLstm = LSTM(100)(winddInput)
winddDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(winddLstm)
winddDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(winddDense_1)

precipbInput = Input(batch_shape=(None, 1, 5))
precipbLstm = LSTM(100)(precipbInput)
precipbDense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(precipbLstm)
precipbDense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(precipbDense_1)

sunFlat = Flatten()(sunInput)

timeInput = Input(shape=(2,))

concat = concatenate([timeInput,sunFlat,tempDense_2,pressDense_2,windsDense_2,precipDense_2,seaDense_2,sunDense_2,humidDense_2,winddDense_2,precipbDense_2])
# concat = concatenate([tempDense_2,pressDense_2])


Dense_1 = Dense(100,activation='relu', kernel_initializer='he_uniform')(concat)
Dense_2 = Dense(100,activation='relu', kernel_initializer='he_uniform')(Dense_1)
Dense_3 = Dense(100,activation='relu', kernel_initializer='he_uniform')(Dense_2)
Dense_4 = Dense(100,activation='relu', kernel_initializer='he_uniform')(Dense_3)
# Dense_5 = Dense(100,activation='relu', kernel_initializer='he_uniform')(Dense_4)
# Dense_6 = Dense(100,activation='relu', kernel_initializer='he_uniform')(Dense_5)




Output = Dense(1,activation='relu', kernel_initializer='he_uniform')(Dense_4)

model = Model(inputs = [tempInput,pressInput,windsInput,precipInput,seaInput,sunInput,humidInput,winddInput,precipbInput,timeInput], outputs = Output)
# model = Model(tempInput, tempDense_2)


model.summary()
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mse'])
```

    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_36 (InputLayer)           (None, 1, 2)         0                                            
    __________________________________________________________________________________________________
    input_31 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_32 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_33 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_34 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_35 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_37 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_38 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    input_39 (InputLayer)           (None, 1, 5)         0                                            
    __________________________________________________________________________________________________
    lstm_28 (LSTM)                  (None, 100)          42400       input_31[0][0]                   
    __________________________________________________________________________________________________
    lstm_29 (LSTM)                  (None, 100)          42400       input_32[0][0]                   
    __________________________________________________________________________________________________
    ...    
    dense_84 (Dense)                (None, 100)          10100       lstm_35[0][0]                    
    __________________________________________________________________________________________________
    dense_86 (Dense)                (None, 100)          10100       lstm_36[0][0]                    
    __________________________________________________________________________________________________
    input_40 (InputLayer)           (None, 2)            0                                            
    __________________________________________________________________________________________________
    flatten_4 (Flatten)             (None, 2)            0           input_36[0][0]                   
    __________________________________________________________________________________________________
    dense_71 (Dense)                (None, 100)          10100       dense_70[0][0]                   
    __________________________________________________________________________________________________
    dense_73 (Dense)                (None, 100)          10100       dense_72[0][0]                   
    ...
    __________________________________________________________________________________________________
    dense_87 (Dense)                (None, 100)          10100       dense_86[0][0]                   
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 904)          0           input_40[0][0]                   
                                                                     flatten_4[0][0]                  
                                                                     dense_71[0][0]                   
                                                                     dense_73[0][0]                   
                                                                     dense_75[0][0]                   
                                                                     dense_77[0][0]                   
                                                                     dense_79[0][0]                   
                                                                     dense_81[0][0]                   
                                                                     dense_83[0][0]                   
                                                                     dense_85[0][0]                   
                                                                     dense_87[0][0]                   
    __________________________________________________________________________________________________
    dense_88 (Dense)                (None, 100)          90500       concatenate_4[0][0]              
    __________________________________________________________________________________________________
    dense_89 (Dense)                (None, 100)          10100       dense_88[0][0]                   
    __________________________________________________________________________________________________
    dense_90 (Dense)                (None, 100)          10100       dense_89[0][0]                   
    __________________________________________________________________________________________________
    dense_91 (Dense)                (None, 100)          10100       dense_90[0][0]                   
    __________________________________________________________________________________________________
    dense_92 (Dense)                (None, 1)            101         dense_91[0][0]                   
    ==================================================================================================
    Total params: 683,101
    Trainable params: 683,101
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
#모델학습단계; 배치사이즈는 3일간의 데이터로 설정
model.fit(x = [temp,press,winds,precip,sea,sun,humid,windd,precip_b,time], y = trainY, epochs=500, batch_size=432)
```

    Epoch 1/500
    4752/4752 [==============================] - 4s 897us/step - loss: 15.2270 - mse: 324.3008
    Epoch 2/500
    4752/4752 [==============================] - 0s 91us/step - loss: 7.0753 - mse: 77.3146
    Epoch 3/500
    4752/4752 [==============================] - 0s 91us/step - loss: 5.1078 - mse: 41.3915
    Epoch 4/500
    4752/4752 [==============================] - 0s 91us/step - loss: 3.7486 - mse: 23.3039
    Epoch 5/500
    4752/4752 [==============================] - 0s 93us/step - loss: 2.9967 - mse: 15.0301
    ...  
    Epoch 497/500
    4752/4752 [==============================] - 0s 101us/step - loss: 0.2817 - mse: 0.1628
    Epoch 498/500
    4752/4752 [==============================] - 0s 104us/step - loss: 0.3211 - mse: 0.1990
    Epoch 499/500
    4752/4752 [==============================] - 0s 100us/step - loss: 0.3262 - mse: 0.2073
    Epoch 500/500
    4752/4752 [==============================] - 0s 99us/step - loss: 0.2833 - mse: 0.1573
    
    <keras.callbacks.callbacks.History at 0x7f28897d1e50>




```python
#모델을 학습데이터에 적용
pred = model.predict([temp,press,winds,precip,sea,sun,humid,windd,precip_b,time])
plt.plot(pred)
plt.plot(trainY)
```




    [<matplotlib.lines.Line2D at 0x7f287f8ff250>]




![png](/assets/image/output_12_1.png)

```
#테스트셋 표준화- 표준화는 모델성능저하로 사용하지않음
for i in ['X00', 'X01', 'X02', 'X03', 'X05', 'X06', 'X07', 'X08', 'X09', 
          'X12', 'X13', 'X15', 'X17', 'X18', 'X20', 'X22', 'X23', 'X24',
          'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33',
          'X35', 'X37', 'X38','X11','X34','X04', 'X10', 'X21', 'X36', 'X39']:
       test[i] = scaler.fit_transform(test[[i]])
```
```python
temp = np.array(test[temperature_cols]).reshape(-1,1,5)
press = np.array(test[pressure_cols]).reshape(-1,1,5)
winds = np.array(test[windspeed_cols]).reshape(-1,1,5)
precip = np.array(test[precipitation_cols]).reshape(-1,1,5)
sea = np.array(test[sealevelpress_cols]).reshape(-1,1,5)
sun = np.array(test[sunshie_cols]).reshape(-1,1,2)
humid = np.array(test[humidity_cols]).reshape(-1,1,5)
windd = np.array(test[winddirection_cols]).reshape(-1,1,5)
precip_b = np.array(test[precip_b_cols]).reshape(-1,1,5)

time1 = np.array([i for i in minute_sin[:144]]*80)
time2 = np.array([i for i in minute_cos[:144]]*80)

time = np.array([(i,j) for i,j in zip(time1,time2)])
```


```python
#모델을 테스트데이터에 적용
pred = model.predict([temp,press,winds,precip,sea,sun,humid,windd,precip_b,time])
plt.plot(pred)
```




    [<matplotlib.lines.Line2D at 0x7f287f54de10>]




![png](/assets/image/output_15_1.png)



```python
#제출형식에 맞게 파일형식 변환
sample_submission = pd.read_csv('sample_submission.csv', index_col='id')
sample_submission["Y18"] = pred

sample_submission.to_csv('submission_s_t_b500.csv')
```
[온도 추정 경진대회]:https://dacon.io/competitions/official/235584/overview/ 