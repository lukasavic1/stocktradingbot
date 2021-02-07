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

SEQ_LEN = 10
EPOCHS = 10
BATCH_SIZE = 8
PERIOD = 14
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED{int(time.time())}"


def preprocess_df(df):
    for col in df.columns:
         if (col!="target" ):
             df[col]=df[col].pct_change()
             df.dropna(inplace=True)
             df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # Ovo ubacuje sve iz jedne vrste osim target-a
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]]) # Prvo je feature a drugo je label

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])
    sequential_data = buys + sells

    X = []
    y = []

    for seq,targets in sequential_data:
        X.append(seq)
        y.append(targets)

    return np.array(X),y


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


pickle_in = open("X.pickle","rb")
df = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
validation_df = pickle.load(pickle_in)


train_x,train_y = preprocess_df(df)
validation_x,validation_y = preprocess_df(validation_df)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

model = Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())


model.add(LSTM(128,input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.0001,decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

history = model.fit(
    train_x,train_y,
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (validation_x,validation_y),
    callbacks = [tensorboard,earlyStopping,checkpoint])