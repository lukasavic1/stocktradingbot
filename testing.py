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

model.load_weights("/content/RNN_Final-03-0.585.hdf5")
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
testing_x,testing_y = preprocess_df(test2)
testing_x = np.asarray(testing_x)
testing_y = np.asarray(testing_y)
scores = model.evaluate(testing_x,testing_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

