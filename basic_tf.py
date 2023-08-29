# # 구글 코랩에서 실행할 경우에만 
# from google.colab import files
# uploaded = files.upload()
# print(uploaded)



#%%

import pandas as pd
import numpy as np
import io


# 파일 불러오기
df_x = pd.read_excel('축구승률.xlsx', sheet_name = '원인')
df_y = pd.read_excel('축구승률.xlsx', sheet_name = '결과')




#%%
# index 변경
df_x = df_x.set_index('항목')
df_y = df_y.set_index('항목')


# O -> 1 변경
# X -> 0 변경
df_x = df_x.replace('O', 1)
df_x = df_x.replace('X', 0)

df_y = df_y.replace('O', 1)
df_y = df_y.replace('X', 0)


# 
varname_x = df_x.index
varname_y = df_y.index

len_x = len(df_x)
len_y = len(df_y)
len_data = len(df_x.columns)


x = np.array(df_x).T
y = np.array(df_y).T




#%%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# 모델 만들기 
model = Sequential()
model.add(Input(shape=(len_x,)))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(len_y, activation = 'Softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mse'])




#%%
# 모델 학습하기
hist = model.fit(x, y, epochs=50, batch_size = 5)



#%%
y_pred = model.predict([[0,0,0,0,0,0,0,0,0,0]])
# y_pred = model.predict(np.expand_dims(x[0], 0))



#%% 결과 출력
for i in range(len_y):
    print(varname_y[i], ' : ', round(y_pred[0][i],3))