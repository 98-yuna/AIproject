#!/usr/bin/env python
# coding: utf-8

# In[21]:


import tensorflow as tf
import keras


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from numpy import hstack
from keras.utils import np_utils
from numpy import array


# In[24]:


# 신경망의 빠른학습을 위한 정규화
def MinMaxScalar(data):
    mise1 = data - np.min(data, 0)
    mise2 = np.max(data, 0) - np.min(data, 0)
    return mise1 / mise2


# In[25]:


feature = 1 # 다음날 미세먼지를 예측하기
steps = 1# 지난날 데이터를 기반으로 하여 


# In[96]:


# 데이터 전처리 
from numpy import array # name 'array' is not defined 오류가 출력됨

def split(mise_data, steps):
    X, y = list(), list()
    
    for i in range(len(mise_data)):
        end = i + steps
        
        if end > len(mise_data)-1:
            break
            
        spl_x, spl_y = mise_data[i:end], mise_data[end]
        X.append(spl_x)
        y.append(spl_y)
        
    return array(X), array(y) # return 형태는 numpy배열형태로 
   


# In[35]:


# 강남구데이터(2010.01.01~2018.12.31)를 Train_set으로 지정하여 데이터 전처리하기 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 저장된 csv 파일을 읽어오기 위해 경로지정
os.chdir("D:/AiProject")

# csv 파일 읽어오기
df = pd.read_csv('Gangnam-training.csv', encoding = "CP949" ) #utf-8오류떠서 인코딩방식지정함
df['PM10'] = df['PM10'].fillna(df['PM10'].mean()).astype(float) # PM10 부분에 공백이 있어서 공백 데이터를 해당 열의 평균으로 대체
df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].mean()).astype(float) # PM 2.5 부분에 공백이 있어서 공백 데이터를 해당 열의 평균으로 대체 


# In[39]:


from numpy import array

# PM 10 열만을 출력하여 배열로 생성 
pm10_data = array(df["PM10"])
# PM 2.5 열만을 출력하여 배열로 생성
pm25_data = array(df["PM2.5"])

x_pm10, y_pm10 = split(pm10_data, steps)
x_pm25, y_pm25 = split(pm25_data, steps)

print(x_pm10.shape)
print(y_pm10.shape)

# 정규화함수를 호출하여 각각의 배열 요소들을 정규화 
a = MinMaxScalar(x_pm10)
b = MinMaxScalar(y_pm10)
c = MinMaxScalar(x_pm25)
d = MinMaxScalar(y_pm25)
# print(a)


# In[77]:


#  LSTM 입력값은 3차원 배열이어야하므로 numpy.reshape함수를 사용하여 2차원 배열을 3차원배열로 형태변환한다.
X_pm10 = a.reshape((x_pm10.shape[0], x_pm10.shape[1], 1))
X_pm25 = a.reshape((x_pm25.shape[0], x_pm25.shape[1], 1))

X_pm10.shape
# RNN에 입력할 훈련셋 생성완료


# In[78]:


# 심층 RNN신경망 LSTM 알고리즘 사용
model = Sequential()
# 입력층 32, 활성화함수 relu를 사용
model.add(LSTM(32, activation = 'relu', input_shape = (steps,feature) )) 

# model.add(LSTM(32, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
# 기계학습
hist1 = model.fit(X_pm10, b, epochs = 50, batch_size = 10 , verbose = 1)

model.summary


# In[79]:

# 오류율 출력하기 
plt.plot(hist1.history['loss'])
plt.ylim(0.0, 0.01)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['PM10'], loc = 'upper right')
plt.show()


# In[89]:


TEST1 = pd.read_csv('Gangnam-test01.csv', encoding = 'CP949') # 테스트셋
TEST1['PM10'] = TEST1['PM10'].fillna(TEST1['PM10'].mean()).astype(float) 


TEST3 = pd.read_csv('Gangnam-test03.csv', encoding = 'CP949') # 테스트셋
TEST3['PM10'] = TEST3['PM10'].fillna(TEST3['PM10'].mean()).astype(float) 


TEST6 = pd.read_csv('Gangnam-tests.csv', encoding = 'CP949') # 테스트셋
TEST6['PM10'] = TEST6['PM10'].fillna(TEST6['PM10'].mean()).astype(float) 

# training set과 마찬가지 방법으로 데이터를 전처리한다. 
test_pm10_1 = array(TEST1['PM10'])
x_testpm10_1, y_testpm10_1 = split(test_pm10_1, steps)
test_a1 = MinMaxScalar(x_testpm10_1)
test_b1 = MinMaxScalar(y_testpm10_1)
Test_PM10_1 = MinMaxScalar(test_a1)
x_testpm10_1= test_a1.reshape((x_testpm10_1.shape[0], x_testpm10_1.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred1 = model.predict(x_testpm10_1)

# training set과 마찬가지 방법으로 데이터를 전처리한다.
test_pm10_3 = array(TEST3['PM10'])
x_testpm10_3, y_testpm10_3 = split(test_pm10_3, steps)
test_a3 = MinMaxScalar(x_testpm10_3)
test_b3 = MinMaxScalar(y_testpm10_3)
Test_PM10_3 = MinMaxScalar(test_a3)
x_testpm10_3= test_a3.reshape((x_testpm10_3.shape[0], x_testpm10_3.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred3 = model.predict(x_testpm10_3)

# training set과 마찬가지 방법으로 데이터를 전처리한다.
test_pm10_6 = array(TEST6['PM10'])
x_testpm10_6, y_testpm10_6 = split(test_pm10_6, steps)
test_a6 = MinMaxScalar(x_testpm10_6)
test_b6 = MinMaxScalar(y_testpm10_6)
Test_PM10_6 = MinMaxScalar(test_a6)
x_testpm10_6 = test_a6.reshape((x_testpm10_6.shape[0], x_testpm10_6.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred6 = model.predict(x_testpm10_6)


# In[90]:


# 2019.01월 , 03월자료를 test set으로 생성하여 예측값과 실제값 비교
# 2019.06.01~ 2019.12.31 자료를 test set으로 만들어 예측값과 실제값 비교 
# 수집한 자료를 보면 가끔씩 비정상적으로 미세먼지가 많이 측정된 날이 보임 평균 50~60이지만 해당 날짜들은 180정도까지 치솟음 

plt.subplot(1,2,1)
plt.plot(Test_PM10_1, 'r')
plt.plot(y_pred1, 'b')
plt.title("Gangnam-gu-Jan")
plt.xlim(0,31)
plt.ylabel('PM10', fontsize = 30)


plt.subplot(1,2,2)
plt.plot(Test_PM10_3, 'r')
plt.plot(y_pred3, 'b')
plt.title("Gangnam-gu-Mar")
plt.xlim(0,31)
plt.ylabel('PM10', fontsize = 30)

# 그래프가 겹치지 않도록 출력하기
plt.tight_layout()
plt.show()


# In[91]:


plt.plot(Test_PM10_6, 'r')
plt.plot(y_pred6, 'b')
plt.title("Gangnam-gu")
plt.ylabel('PM10', fontsize = 30)
plt.show()


# In[92]:


# 예측값과 실제값 오류를 알아보기
# mse 함수를 사용하여 오차율을 계산한다

from sklearn.metrics import mean_squared_error
from math import sqrt

error1 = sqrt(mean_squared_error(Test_PM10_1, y_pred1))
error2 = sqrt(mean_squared_error(Test_PM10_3, y_pred3))
error3 = sqrt(mean_squared_error(Test_PM10_6, y_pred6))

print("2019년 1월 미세먼지예측값 오차율 = ", error1)
print("2019년 3월 미세먼지예측값 오차율 = ", error2)
print("2019년 하반기 미세먼지예측값 오차율 = ", error3)


# In[84]:


# 초미세먼지 학습시키기 
# 위의 미세먼지 학습과 동일한 방식을 사용한다
model2 = Sequential()
model2.add(LSTM(32, activation = 'relu', input_shape = (steps,feature) )) 
model2.add(Dense(1))
model2.compile(optimizer = 'adam', loss='mse')

hist2 = model2.fit(X_pm25, c, epochs = 50, batch_size = 10, verbose = 1)
model.summary


# In[93]:


TEST1 = pd.read_csv('Gangnam-test01.csv', encoding = 'CP949') # 테스트셋
TEST1['PM2.5'] = TEST1['PM2.5'].fillna(df['PM2.5'].mean()).astype(float) 


TEST3 = pd.read_csv('Gangnam-test03.csv', encoding = 'CP949') # 테스트셋
TEST3['PM2.5'] = TEST3['PM2.5'].fillna(df['PM2.5'].mean()).astype(float) 


TEST6 = pd.read_csv('Gangnam-tests.csv', encoding = 'CP949') # 테스트셋
TEST6['PM2.5'] = TEST6['PM2.5'].fillna(df['PM2.5'].mean()).astype(float) 

# 위의 training set 데이터 전처리 방식과 동일 
test_pm25_1 = array(TEST1['PM2.5'])
x_testpm25_1, y_testpm25_1 = split(test_pm25_1, steps)
test_a1 = MinMaxScalar(x_testpm25_1)
test_b1 = MinMaxScalar(y_testpm25_1)
Test_PM25_1 = MinMaxScalar(test_a1)
x_testpm25_1= test_a1.reshape((x_testpm25_1.shape[0], x_testpm25_1.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred1 = model.predict(x_testpm25_1)

# 위의 training set 데이터 전처리 방식과 동일 
test_pm25_3 = array(TEST3['PM2.5'])
x_testpm25_3, y_testpm25_3 = split(test_pm25_3, steps)
test_a3 = MinMaxScalar(x_testpm25_3)
test_b3 = MinMaxScalar(y_testpm25_3)
Test_PM25_3 = MinMaxScalar(test_a3)
x_testpm25_3= test_a3.reshape((x_testpm25_3.shape[0], x_testpm25_3.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred3 = model.predict(x_testpm25_3)

# 위의 training set 데이터 전처리 방식과 동일 
test_pm25_6 = array(TEST6['PM2.5'])
x_testpm25_6, y_testpm25_6 = split(test_pm25_6, steps)
test_a6 = MinMaxScalar(x_testpm25_6)
test_b6 = MinMaxScalar(y_testpm25_6)
Test_PM25_6 = MinMaxScalar(test_a6)
x_testpm25_6 = test_a6.reshape((x_testpm25_6.shape[0], x_testpm25_6.shape[1], 1))

# 테스트 모델을 가지고 예측해보기
y_pred6 = model.predict(x_testpm25_6)


# In[94]:

plt.subplot(1,2,1)
plt.plot(Test_PM25_1, 'r')
plt.plot(y_pred1, 'b')
plt.title("Gangnam-gu-Jan")
plt.xlim(0,31)
plt.ylabel('PM2.5', fontsize = 30)

plt.subplot(1,2,2)
plt.plot(Test_PM25_3, 'r')
plt.plot(y_pred3, 'b')
plt.title("Gangnam-gu-Mar")
plt.xlim(0,31)
plt.ylabel('PM2.5', fontsize = 30)

# 그래프 겹치지 않도록 출력하기
plt.tight_layout()
plt.show()


# In[95]:

plt.plot(Test_PM25_6, 'r')
plt.plot(y_pred6, 'b')
plt.title("Gangnam-gu")
plt.ylabel('PM2.5', fontsize = 30)
plt.show()


# In[88]:

# 예측값과 실제 값 오류를 알아보기
# mse 함수 사용 
from sklearn.metrics import mean_squared_error
from math import sqrt

error1 = sqrt(mean_squared_error(Test_PM25_1, y_pred1))
error2 = sqrt(mean_squared_error(Test_PM25_3, y_pred3))
error3 = sqrt(mean_squared_error(Test_PM25_6, y_pred6))

print("2019년 1월 초미세먼지예측값 오차율 = ", error1)
print("2019년 3월 초미세먼지예측값 오차율 = ", error2)
print("2019년 하반기 초미세먼지예측값 오차율 = ", error3)

