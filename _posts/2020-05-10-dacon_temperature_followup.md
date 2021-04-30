---
layout: single
title: "데이콘 기상센서 활용 온도추정 대회 팔로업"
date: 2020-05-10
comments: true
categories: 
    [PROJECT]
tags:
    [Competition, Code Review]
toc: false
publish: true
classes: wide
comments: true
---

데이콘에서 열린 기상센서 활용 온도추정 대회 결과에 대해 시상식 및 수상작 발표회를 봤습니다. 본 대회에서는 992개의 참가팀 중에서 41등을 달성했는데 1,2,3등을 하신분들의 프레젠테이션을 보니 아직 배우고 실험해야되는 과정이 많이 남아있다는것을 깨닫게되었습니다.

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/dx50qY6Y8Lw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

새롭게 배운점을 레퍼런스를 위해 간략하게 요약해보았습니다:

3등_모델깎는 노인님:
* EDA과정을 통해 시간대별 온도차이는 있으나 전체적인 흐름은 유사함을 확인했고 해면기압과 지면기압 변수에 대해 온도에 직접적인 영향이 적다고 판단했습니다. EDA 과정은 모델생성이전에 데이터의 특징을 파악하기위해 중요한 과정이라는것을 다시 알 수 있었습니다.
* 풍향과 같은 변수는 데이터로서 의미보다는 방향으로서의 의미가 중요하다고 판단해 원핫인코딩을 했습니다.
* 누적일사량과 누적강수량에 대해 10분단위의 변화량에 초점을 두었습니다. 이걸 30분 단위, 1시간 단위로 확장했네요.

```python
X_data_in_diff_y = X_data
X_data_in_diff_y['Insol_1_10min'] = X_data_in_diff_y['Insol_1'].diff()
X_data_in_diff_y['Insol_5_10min'] = X_data_in_diff_y['Insol_5'].diff()

X_data_in_diff_y

Insol_10min = ['Insol_1_10min','Insol_5_10min']
Rain_10min = ['Rain_1_10min','Rain_2_10min','Rain_3_10min','Rain_4_10min','Rain_5_10min']

for i in Insol_10min:
    X_data_in_diff_y.loc[X_data_in_diff_y.query('minute ==0').index,i] = 0
X_data_diff_y = X_data_in_diff_y

X_data_in_diff_y['Rain_1_10min'] = X_data_in_diff_y['Rain_1'].diff()
X_data_in_diff_y['Rain_2_10min'] = X_data_in_diff_y['Rain_2'].diff()
X_data_in_diff_y['Rain_3_10min'] = X_data_in_diff_y['Rain_3'].diff()
X_data_in_diff_y['Rain_4_10min'] = X_data_in_diff_y['Rain_4'].diff()
X_data_in_diff_y['Rain_5_10min'] = X_data_in_diff_y['Rain_5'].diff()
```

* 논문을 참고해 많은 파생변수들을 생성했습니다: 이슬점, 공기밀도, 체감온도, Summer Simmer Index, 고도, 관측소별 강수량 평균, 강수량의 유무, 증기압, 혼합비, 일자별 일교차, 일자별 평균온도, 등등. 대단합니다.
* 모델과 같은 경우 GB랜덤서치, XGB랜덤서치, LGBM 랜덤서치로 나온 값에 G-MEAN을 통해 기하평균값을 도출해냈습니다. 모델자체가 큰 사이즈이지만 Dropout값과 같은 규제를 통해 과적합을 방지한게 더 좋은 결과를 만들어낼 수 있었다고 합니다.


1등_최상혁님:
* 위의 참가자분과 비슷하게 누적량을 빼고 정해진 시간동안에 일사량과 강수량을 구하셨습니다. 저도 대회 중에 적용하고싶었던 부분이였으나 개념에 대한 코드구현에서 어려움을 겪었던 부분입니다.

```python
#Time Series Data의 특성을 활용하기 위해 Date 와 Time 변수를 먼저 생성합니다.
for X in X_list + [combined_X]:
    X['date'] = X.index // 144
    X['time'] = X.index % 144

# 일일 누적치는 실시간 온도 추정에서 설명력이 없으므로 정해진 시간 동안의 Insolation & Precipitation을 구함.

# Modify the insolation (1시간) : 원래 값에서 1시간 이전(periods=6)의 값을 뺀 것으로 대치. 
# 일사량은 일자가 바뀌는 자정 부근에 0이므로, shift로 인한 null 값은 0으로 fill.
for X in X_list:
    dates = X['date'].unique()
    for d in dates:
        day_idx = X.loc[X['date'] == d].index
        for f in indicator_dict['insolation']:
            X.loc[day_idx, f] = X.loc[day_idx, f] - X.loc[day_idx, f].shift(periods=6).fillna(0)

# Modify the Precipitation (6시간) : 원래 값에서 3시간 이전(periods=18)의 값을 뺀 것으로 대치. 
# 강수량은 시간대와 관계없이 나타나므로, 날짜가 바뀌는 날에 0으로 reset됨. => 하루 전날의 누적 강수량을 더하여 값을 구함.
for X in X_list:
    dates = X['date'].unique()
    for f in indicator_dict['precipitation']:
        for d in dates[::-1]:
            day_idx = combined_X.loc[combined_X['date'] == d].index
            yes_day_idx = combined_X.loc[(combined_X['date'] == d-1)].index
            day_precip = combined_X.loc[(yes_day_idx | day_idx), f]
            
            # 편의를 위해 Train1, Train2, Test set 각각의 첫날(d=0)의 하루 전 누적 강수량은 0으로 고정함.
            yesterday_cum_precip = day_precip.loc[yes_day_idx].max() if d != 0 else 0 
            day_precip.loc[day_idx] += yesterday_cum_precip
            day_precip_shift = day_precip.shift(periods=36).fillna(0)
            
            X.loc[day_idx, f] = day_precip.loc[day_idx] - day_precip_shift.loc[day_idx]
```

* EDA과정을 통해 일일 최고 기온에 도달한뒤 약 1시간 정도 후에 기온이 급강하 하는 패턴을 찾으셨는데 이를 관측소의 지리적 요인에 의해 그림자가 생긴것으로 가설을 세우고 neg_insol이라는 새로운 변수를 생성하셔서 문제를 해결했습니다.
* 변수의 수가 많지 않기 때문에 각 변수들을 하나씩 제거해가며 공개점수를 통한 모델 성능 비교를 하셨습니다. 
* 모델 생성에 있어서 일사량은 낮 시간대의 온도에만 영향을 미치기 때문에 낮과 밤을 나누어서 각각 모델을 생성하는 방향으로 하셨습니다: 각각 Train의 X->Y(Y00-Y17) Mapping 을 LASSO 학습해 새로운 트래인 (Y18)과 테스트의 Y를 예측

코드에 대한 자세한 설명과 원문은 [여기서](https://dacon.io/competitions/official/235584/codeshare/) 확인하실 수 있습니다.