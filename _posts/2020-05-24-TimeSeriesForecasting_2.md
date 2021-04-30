---
layout: single
title: "텐서플로우 시계열 예측_파트2_다변량"
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

텐플 튜토리올의 두번째 파트는 다변량 데이터를 사용한 시계열 예측입니다.

제공된 데이터셋에는 총 14가지의 변수가 존재하지만 튜토리얼 과정의 편리를 위해 이 중 온도, 대기압, 공기밀도 3가지만 사용하겠습니다.


```python
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
```


```python
features = df[features_considered]
features.index = df['Date Time']
features.head()
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
      <th>p (mbar)</th>
      <th>T (degC)</th>
      <th>rho (g/m**3)</th>
    </tr>
    <tr>
      <th>Date Time</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>01.01.2009 00:10:00</td>
      <td>996.52</td>
      <td>-8.02</td>
      <td>1307.75</td>
    </tr>
    <tr>
      <td>01.01.2009 00:20:00</td>
      <td>996.57</td>
      <td>-8.41</td>
      <td>1309.80</td>
    </tr>
    <tr>
      <td>01.01.2009 00:30:00</td>
      <td>996.53</td>
      <td>-8.51</td>
      <td>1310.24</td>
    </tr>
    <tr>
      <td>01.01.2009 00:40:00</td>
      <td>996.51</td>
      <td>-8.31</td>
      <td>1309.19</td>
    </tr>
    <tr>
      <td>01.01.2009 00:50:00</td>
      <td>996.51</td>
      <td>-8.27</td>
      <td>1309.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
features.plot(subplots=True)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000001BA04EFCAC8>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x000001BA04FC3D08>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x000001BA04F73F48>],
          dtype=object)




![png](/assets/image/output_38_1.png)



```python
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset - data_mean)/data_std
```

## 싱글 스탭 모델 (Single Step Model)

싱글 스탭 과정을 통해 과거 데이터들을 바탕으로 미래의 데이터 포인트 한점을 예측할 수 있습니다. 다음에 보여드릴 부분은 윈도우 생성을 통해 과거 데이터를 스탭 사이즈에 따라 수집하는 방법입니다.


```python
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    
    return np.array(data), np.array(labels)
```


```python
past_history = 720 #지난 5일동안 수집된 데이터
future_target = 72 #12시간 앞선값을 예측
STEP = 6 #한시간 간격으로 샘플링

x_train_single, y_train_single = multivariate_data(dataset, dataset[:,1],0,
                                                  TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:,1],
                                              TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)
```


```python
print('Single window of past history: {}'. format(x_train_single[0].shape))
```

    Single window of past history: (120, 3)
    


```python
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
```


```python
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
```


```python
for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)
```

    (256, 1)
    


```python
single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                           steps_per_epoch=EVALUATION_INTERVAL,
                                           validation_data=val_data_single,
                                           validation_steps=50)
```

    Train for 200 steps, validate for 50 steps
    Epoch 1/10
    200/200 [==============================] - 26s 130ms/step - loss: 0.2978 - val_loss: 0.2584
    Epoch 2/10
    200/200 [==============================] - 28s 139ms/step - loss: 0.2612 - val_loss: 0.2454
    Epoch 3/10
    200/200 [==============================] - 32s 161ms/step - loss: 0.2594 - val_loss: 0.2396
    Epoch 4/10
    200/200 [==============================] - 37s 184ms/step - loss: 0.2569 - val_loss: 0.2392
    Epoch 5/10
    200/200 [==============================] - 43s 214ms/step - loss: 0.2247 - val_loss: 0.2344
    Epoch 6/10
    200/200 [==============================] - 49s 244ms/step - loss: 0.2397 - val_loss: 0.2653
    Epoch 7/10
    200/200 [==============================] - 53s 266ms/step - loss: 0.2417 - val_loss: 0.2532
    Epoch 8/10
    200/200 [==============================] - 58s 290ms/step - loss: 0.2407 - val_loss: 0.2460
    Epoch 9/10
    200/200 [==============================] - 61s 304ms/step - loss: 0.2410 - val_loss: 0.2437
    Epoch 10/10
    200/200 [==============================] - 63s 315ms/step - loss: 0.2384 - val_loss: 0.2341
    


```python
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()
```


```python
plot_train_history(single_step_history,
                  'Single Step Training and Validation Loss')
```


![png](/assets/image/output_49_0.png)


## 싱글 스탭을 통한 예측

이제 모델을 학습했으므로 몇몇 예측값을 실험해보겠습니다. 모델은 5일간에 수집된 3가지 변수 데이터로 학습되었으며 이는 총 120개의 데이터 포인트에 달합니다. 예측하고자하는 것이 오직 온도이기 때문에 그래프 상에서는 온도만 표기하도록 하겠습니다. 


```python
for x,y in val_data_single.take(3):
    plot = show_plot([x[0][:,1].numpy(), y[0].numpy(),
                     single_step_model.predict(x)[0]],12,
                    'Single Step Prediction')
    plot.show()
```


![png](/assets/image/output_51_0.png)



![png](/assets/image/output_51_1.png)



![png](/assets/image/output_51_2.png)


## 멀티 스탭 모델

멀티 스탭 모델에서는 과거의 데이터를 바탕으로 미래의 한 데이터 포인트가 아닌 연속적인 값을 예측해야합니다. 따라서 싱글 스탭 모델과는 달리 멀티 스탭 모델에서는 연속적인 값을 예측하게됩니다.

학습을 위한 데이터는 과거 5일간 매시간 마다 수집되었던 데이터를 활용했습니다. 하지만 여기서는 모델을 활용해 미래의 12시간에 해당되는 값을 예측하려고합니다. 각각의 데이터 수집은 10분간격으로 이루어지니 최종 아웃풋은 72개의 데이터 포인트로 구성될 것입니다. 싱글 스탭 과정과 유사하면서도 이러한 차이를 반영하기 위해 다른 타겟 윈도우 값을 설정할 예정입니다


```python
future_target = 72
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                TRAIN_SPLIT, past_history,
                                                future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                            TRAIN_SPLIT, None, past_history,
                                            future_target, STEP)
```


```python
print('Single window of past history: {}'. format(x_train_multi[0].shape))
print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))
```

    Single window of past history: (120, 3)
    
     Target temperature to predict : (72,)
    


```python
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
```


```python
def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12,6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    
    plt.plot(num_in, np.array(history[:,1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future),'bo', label='True Future')
    
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
```


```python
for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))
```


![png](/assets/image/output_57_0.png)


예측 테스크가 기존보다 복잡해졌기 때문에 모델에 LSTM 레이어를 추가할 필요가 있습니다. 72가지 데이터 포인트가 결과로 나와야하기 때문에 DENSE 레이어 아웃풋 또한 72개의 예측값을 담을 수 있게 설정이 필요합니다.


```python
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                         return_sequences = True,
                                         input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
```


```python
for x,y in val_data_multi.take(1):
    print(multi_step_model.predict(x).shape)
```

    (256, 72)
    


```python
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)
```

    Train for 200 steps, validate for 50 steps
    Epoch 1/10
    200/200 [==============================] - 81s 407ms/step - loss: 3.4699 - val_loss: 0.3256
    Epoch 2/10
    200/200 [==============================] - 88s 441ms/step - loss: 0.3566 - val_loss: 0.2854
    Epoch 3/10
    200/200 [==============================] - 104s 520ms/step - loss: 0.3569 - val_loss: 0.2663
    Epoch 4/10
    200/200 [==============================] - 125s 624ms/step - loss: 0.2625 - val_loss: 0.2152
    Epoch 5/10
    200/200 [==============================] - 158s 789ms/step - loss: 0.2088 - val_loss: 0.2063
    Epoch 6/10
    200/200 [==============================] - 171s 855ms/step - loss: 0.2144 - val_loss: 0.2100
    Epoch 7/10
    200/200 [==============================] - 169s 847ms/step - loss: 0.2031 - val_loss: 0.2169
    Epoch 8/10
    200/200 [==============================] - 173s 865ms/step - loss: 0.1979 - val_loss: 0.2004
    Epoch 9/10
    200/200 [==============================] - 181s 907ms/step - loss: 0.2012 - val_loss: 0.1873
    Epoch 10/10
    200/200 [==============================] - 184s 919ms/step - loss: 0.1931 - val_loss: 0.1946
    


```python
plot_train_history(multi_step_history, 'Multi-Step Training and Validation Loss')
```


![png](/assets/image/output_62_0.png)



```python
for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
```


![png](/assets/image/output_63_0.png)



![png](/assets/image/output_63_1.png)



![png](/assets/image/output_63_2.png)

