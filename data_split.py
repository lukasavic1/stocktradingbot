import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
def split_df_pct(df,pct):
    length = int(len(df)*pct)
    list1 = []
    list2 = []
    for i in range(0,length):
        list1.append(df.iloc[i])
    for i in range(length,len(df)):
        list2.append(df.iloc[i])
    return pd.DataFrame(list1),pd.DataFrame(list2)

def split_df_3(df):
    length = int(len(df)/3)
    list1 = []
    list2 = []
    list3 = []
    for i in range(0,length):
        list1.append(df.iloc[i])
    for i in range(length,2*length):
        list2.append(df.iloc[i])
    for i in range(2*length,3*length):
        list3.append(df.iloc[i])
    return pd.DataFrame(list1),pd.DataFrame(list2),pd.DataFrame(list3)

df,testing_df = split_df_pct(df,0.9)
df,validation_df = split_df_pct(df,0.9)


print(len(df.index))
print('Train data')
print(df.head())
print(len(df.index))
print('Validation data')
print(validation_df.head())
print(len(validation_df.index))
print('Test data')
print(testing_df.head())
print(len(testing_df.index))
print(len(testing_df))


test1,test2,test3 = split_df_3(testing_df)
print('----------------------')
print(len(test1.index))
print(test1.head())
print(len(test2.index))
print(test2.head())
print(len(test3.index))
print(test3.head())

pickle_out = open("X.pickle","wb")
pickle.dump(df,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(validation_df,pickle_out)
pickle_out.close()