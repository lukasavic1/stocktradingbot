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
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def df_ATR(df,period):
    lst = []
    lst.append(0.0001)
    prev_days_ATR = deque(maxlen=period)
    for i in range(1,len(df.index)):
        time_0, open_0, high_0, low_0, close_0,close_adjusted_0, volume_0 = df.iloc[i-1]
        time, open, high, low, close, close_adjusted, volume = df.iloc[i]
        tr = max(abs(high-low),abs(high-close_0),abs(close_0-low))
        prev_days_ATR.append(tr)
        ATR = 0.001
        count = 0
        if(len(prev_days_ATR)==period):
            for ct in range(0,period):
                ATR = ATR + prev_days_ATR[ct]
            ATR = 1/period * ATR
        else:
            for ct in range(0, len(prev_days_ATR)):
                ATR = ATR + prev_days_ATR[ct]
                count = count + 1
            ATR = ATR/count
        lst.append(ATR)
    return lst

def df_SMA(df,period):
    lst = []
    prev_days_SMA = deque(maxlen=period)
    for i in range(0,len(df.index)):
        time, open, high, low, close, close_adjusted, volume, atr = df.iloc[i]
        prev_days_SMA.append(close)
        SMA = 0
        if(len(prev_days_SMA)==period):
            for ct in range(0,period):
                SMA = SMA + prev_days_SMA[ct]
            SMA = 1/period * SMA
        else:
            SMA = close
        lst.append(SMA)
    return lst

def df_RSI(df,PERIOD):
    lst = []
    lst.append(50)
    counter = 0
    previous_average_gain=0
    previous_average_loss=0
    first_average_gain = 0
    first_average_loss = 0
    for i in range(1,len(df.index)):
        time, open, high, low, close, close_adjusted, volume, atr, sma = df.iloc[i]
        res = close - open
        if res>=0:
            res_gain = res
            res_loss = 0
        else:
            res_gain = 0
            res_loss = abs(res)
        counter+=1
        if counter<14:
            if(res>=0):
                first_average_gain+=res
            else:
                first_average_loss+=res
        if counter==14:
            first_average_loss=abs(first_average_loss)/PERIOD
            first_average_gain=abs(first_average_gain)/PERIOD
            previous_average_gain = first_average_gain
            previous_average_loss = first_average_loss
        if counter>=14:
            average_gain = ((PERIOD-1)*previous_average_gain+res_gain)/PERIOD
            average_loss = ((PERIOD-1)*previous_average_loss+res_loss)/PERIOD
            RS = average_gain/average_loss
            RSI = 100 - (100/(1+RS))
            if(RSI==0):
               RSI=0.001
            previous_average_gain = average_gain
            previous_average_loss = average_loss
        if counter<14:
            RSI = 50
        lst.append(RSI)
    return lst

def df_1RR(df):
    lst = []
    strategy_atr = 2
    for i in range(1,len(df.index)):
        time, open, high, low, close, close_adjusted, volume, atr, sma, rsi = df.iloc[i]
        trade_price = close
        trade_atr = atr
        positive_change = 0
        negative_change = 0
        j = 0
        while(positive_change<strategy_atr*trade_atr and negative_change<strategy_atr*trade_atr):
            j += 1
            if((j+i)>=len(df.index)):
                lst.append(2)
                break
            time_0, open_0, high_0, low_0, close_0, close_adjusted_0, volume_0, atr_0, sma_0, rsi_0 = df.iloc[i+j]
            positive_change = high_0 - trade_price
            negative_change = trade_price - low_0
        if(positive_change>=strategy_atr*trade_atr):
            lst.append(1) # 1 means buy
        elif(negative_change>=strategy_atr*trade_atr):
            lst.append(0) # 0 means sell
    return lst


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

PERIOD = 14

df = pd.read_csv("/content/gdrive/MyDrive/appl_yahoofinance.csv",names=["time","open","high","low","close","adjusted_close","volume"])
print(df.size)

df['ATR {period}'.format(period = PERIOD)] = pd.DataFrame(df_ATR(df,PERIOD))
print("ATR Done")
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
df['SMA {period}'.format(period = 14)] = pd.DataFrame(df_SMA(df,14))
print('SMA done')
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
df['RSI {period}'.format(period=PERIOD)] = pd.DataFrame(df_RSI(df,PERIOD))
print("RSI Done")
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
df['target'] = pd.DataFrame(df_1RR(df))
df['target'] = df['target'].shift(1)
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
print("Target Done")
df = df.shift(20)
df.dropna(inplace=True)
df.set_index("time", inplace=True)
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
atr_name = "ATR {period}".format(period=PERIOD)
rsi_name = "RSI {period}".format(period = PERIOD)
sma14_name = "SMA {period}".format(period = 14)
df = df[["close","volume",atr_name,sma14_name,rsi_name,"target"]]
df.dropna(inplace=True)