---
layout: single
title: "텐서플로우 시계열 예측_파트1_단변량"
date: 2020-05-24
comments: true
categories: 
    [Time-Series]
tags:
    [LSTM, Climate Data, Univariate, RNN]
toc: false
publish: true
classes: wide
comments: true
---

### 텐서플로우 공식 가이드 튜토리얼
## 시계열 예측 (Time Series Forecasting)
- ohikendoit.github.io 번역
- [구글 코랩 버전](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb), [깃헙 소스코드](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb), [주피터 노트북 버전](https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/structured_data/time_series.ipynb)

본 튜토리얼을 통해 Recurrent Neural Network (순환신경망)을 활용한 시계열 예측을 해볼 수 있습니다. 모델 생성은 총 두가지 파트로 나누어져있는데 첫번째는 단일변량, 두번째는 다변량 시계열 데이터를 가지고 실험하게됩니다. 여기서는 먼저 간단하게 단일 변량 데이터를 다루어보겠습니다.


```python
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False
```

사용하게될 데이터는 독일에 있는 막스 플랑크 화학 연구소 (Max Planck Institute for Biogeochemistry) 에서 공개한 시계열 날씨 데이터셋입니다.

해당 데이터셋은 온도, 기압, 습도를 포함한 14가지 변수를 담고있습니다. 막스 플랑크 연구소에서는 2003년도 부터 10분간격으로 측정을 하고 있으며, 이 튜토리올에서는 2009년부터 2016년 동안 수집된 데이터만을 활용했습니다. Francois Chollet 저자의 Deep Learning with Python 이라는 책을 참고하여 데이터 전처리를 진행했습니다. 


```python
zip_path = tf.keras.utils.get_file(
    origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname = 'jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```


```python
df = pd.read_csv(csv_path)
```


```python
df.head()
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
      <th>Date Time</th>
      <th>p (mbar)</th>
      <th>T (degC)</th>
      <th>Tpot (K)</th>
      <th>Tdew (degC)</th>
      <th>rh (%)</th>
      <th>VPmax (mbar)</th>
      <th>VPact (mbar)</th>
      <th>VPdef (mbar)</th>
      <th>sh (g/kg)</th>
      <th>H2OC (mmol/mol)</th>
      <th>rho (g/m**3)</th>
      <th>wv (m/s)</th>
      <th>max. wv (m/s)</th>
      <th>wd (deg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>01.01.2009 00:10:00</td>
      <td>996.52</td>
      <td>-8.02</td>
      <td>265.40</td>
      <td>-8.90</td>
      <td>93.3</td>
      <td>3.33</td>
      <td>3.11</td>
      <td>0.22</td>
      <td>1.94</td>
      <td>3.12</td>
      <td>1307.75</td>
      <td>1.03</td>
      <td>1.75</td>
      <td>152.3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>01.01.2009 00:20:00</td>
      <td>996.57</td>
      <td>-8.41</td>
      <td>265.01</td>
      <td>-9.28</td>
      <td>93.4</td>
      <td>3.23</td>
      <td>3.02</td>
      <td>0.21</td>
      <td>1.89</td>
      <td>3.03</td>
      <td>1309.80</td>
      <td>0.72</td>
      <td>1.50</td>
      <td>136.1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>01.01.2009 00:30:00</td>
      <td>996.53</td>
      <td>-8.51</td>
      <td>264.91</td>
      <td>-9.31</td>
      <td>93.9</td>
      <td>3.21</td>
      <td>3.01</td>
      <td>0.20</td>
      <td>1.88</td>
      <td>3.02</td>
      <td>1310.24</td>
      <td>0.19</td>
      <td>0.63</td>
      <td>171.6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>01.01.2009 00:40:00</td>
      <td>996.51</td>
      <td>-8.31</td>
      <td>265.12</td>
      <td>-9.07</td>
      <td>94.2</td>
      <td>3.26</td>
      <td>3.07</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>3.08</td>
      <td>1309.19</td>
      <td>0.34</td>
      <td>0.50</td>
      <td>198.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>01.01.2009 00:50:00</td>
      <td>996.51</td>
      <td>-8.27</td>
      <td>265.15</td>
      <td>-9.04</td>
      <td>94.1</td>
      <td>3.27</td>
      <td>3.08</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>3.09</td>
      <td>1309.00</td>
      <td>0.32</td>
      <td>0.63</td>
      <td>214.3</td>
    </tr>
  </tbody>
</table>
</div>



데이터 프레임으로부터 볼 수 있듯이 10분 간격으로 수집된 데이터 포인트가 존재합니다. 이는 6개의 데이터 포인트가 모이면 한시간을 뜻하는것과 같습니다. 그러므로 하루에는 144개 (6x24) 의 데이터 포인트로 구성되어있습니다.

예를들어 6시간후 온도를 예측한다고 할때, 지난 720개 (5x144)의 데이터 포인트를 활용한다면 5일 동안 수집된 정보를 모델 학습에 활용할 수 있습니다. 데이터 셋에는 다양한 변수들이 있기 때문에 이들의 조합을 통해 모델 성능 향상을 실험해 볼  수 있습니다.

다음 소개해드릴 'univariate_data' 함수는 방금 말씀드렸던 모델 학습에 적용할 타임 프레임(Windows of Time)을 정하는 함수입니다. 변수로 언급된 history_size는 타임프레임의 크기이며, target_size는 모델을 통해 예측할 기간의 크기라고 생각하시면 됩니다. 다른 말로 target_size는 예측이 필요한 레이블이라고 볼 수 있습니다.


```python
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data=[]
    labels=[]
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        #Reshape data from (history_size,) to (history_size,1)
        data.append(np.reshape(dataset[indices], (history_size,1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)
```

본 튜토리얼에서는 모델 학습을 위해 300,000개의 행을 사용하고 나머지 데이터는 검증을 위해서만 사용하도록 하겠습니다. 이는 거의 2100일에 가까운 학습데이터입니다.


```python
TRAIN_SPLIT = 300000
tf.random.set_seed(13)
```

## 파트1: 단일변량 시계열 데이터 예측


```python
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()
```




    Date Time
    01.01.2009 00:10:00   -8.02
    01.01.2009 00:20:00   -8.41
    01.01.2009 00:30:00   -8.51
    01.01.2009 00:40:00   -8.31
    01.01.2009 00:50:00   -8.27
    Name: T (degC), dtype: float64




```python
uni_data.plot(subplots=True)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000001BA6F44C6C8>],
          dtype=object)




![png](/assets/image/output_13_1.png)



```python
uni_data = uni_data.values
```

뉴럴 네트워크 모델을 학습하기 전에 각각의 변수들의 비율(Scale)을 맞추는것이 중요합니다.
 
데이터 값에 평균값을 빼고 표준편차를 나누는 데이터 표준화 (Standardization) 과정은 이를 달성하기 위해 가장 많이 사용되는 방식입니다. 케라스 페키지에는 tf.keras.utils.normalize 기능이 포함되어있어 이를 사용하면 데이터값을 [0,1] 범위 안으로 자동적으로 축적하게됩니다.


```python
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
```


```python
uni_data = (uni_data - uni_train_mean)/(uni_train_std)
```

모델 생성에 있어서 가장 최근에 수집된, 마지막 20개의 데이터포인트를 사용해서 미래 온도를 예측해보겠습니다.


```python
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, univariate_past_history, univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
```


```python
print('Single window of past history')
print(x_train_uni[0])
print('\n Target temperature to predict')
print(y_train_uni[0])
```

    Single window of past history
    [[-1.99766294]
     [-2.04281897]
     [-2.05439744]
     [-2.0312405 ]
     [-2.02660912]
     [-2.00113649]
     [-1.95134907]
     [-1.95134907]
     [-1.98492663]
     [-2.04513467]
     [-2.08334362]
     [-2.09723778]
     [-2.09376424]
     [-2.09144854]
     [-2.07176515]
     [-2.07176515]
     [-2.07639653]
     [-2.08913285]
     [-2.09260639]
     [-2.10418486]]
    
     Target temperature to predict
    -2.1041848598100876
    


```python
def create_time_steps(length):
    return list(range(-length, 0))
```


```python
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0
        
    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt
```


```python
show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
```




    <module 'matplotlib.pyplot' from 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\pyplot.py'>




![png](/assets/image/output_23_1.png)


아래 나온 베이스라인 모델에서는 모든 과거 데이터 포인트 중 가장 최근에 발생한 20개의 포인트의 평균을 계산하게됩니다.
단순한 통계를 활용한 결과는 당연하게도 정확하지 않아 보이는군요.


```python
def baseline(history):
    return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')
```




    <module 'matplotlib.pyplot' from 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\pyplot.py'>




![png](/assets/image/output_25_1.png)


## 순환신경망(Recurrent Neural Network)

순환신경망을 통해 시계열 데이터를 한 시점 한시점씩 처리할 수 있으며, 이는 해당 시점까지 인풋으로 들어갔던 데이터를 요약하는 성능으로 이어집니다. 더 자세한 내용을 위해서는 다음 링크를 참고하면 되고 [RNN tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent), 시계열에 특화된 순환신경망 레이어인 [LSTM](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/LSTM)에 대해서 읽어볼 수 있다




```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

```

BATCH사이즈로 변경을하면 다음과 같은 모양으로 데이터 모양이 생성된다:

<img src="https://www.tensorflow.org/tutorials/structured_data/images/time_series.png" width="400">


```python
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
```


```python
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)
```

    (256, 1)
    


```python
EVALUATION_INTERVAL = 200 #데이터셋 크기가 큼으로 시간을 절약하기위해서 각 EPOCH는 200스탭만 실시합니다
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_univariate, validation_steps=50)
```

    Train for 200 steps, validate for 50 steps
    Epoch 1/10
    200/200 [==============================] - 4s 18ms/step - loss: 0.4075 - val_loss: 0.1351
    Epoch 2/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.1118 - val_loss: 0.0359
    Epoch 3/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0489 - val_loss: 0.0290
    Epoch 4/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0443 - val_loss: 0.0258
    Epoch 5/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0299 - val_loss: 0.0235
    Epoch 6/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0317 - val_loss: 0.0224
    Epoch 7/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0286 - val_loss: 0.0207
    Epoch 8/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0263 - val_loss: 0.0195
    Epoch 9/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0253 - val_loss: 0.0181
    Epoch 10/10
    200/200 [==============================] - 2s 10ms/step - loss: 0.0227 - val_loss: 0.0174
    




    <tensorflow.python.keras.callbacks.History at 0x1ba6fcd9548>




```python
for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
```


![png](/assets/image/output_33_0.png)



![png](/assets/image/output_33_1.png)



![png](/assets/image/output_33_2.png)

