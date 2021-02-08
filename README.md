# Stock Trading Bot

This repository contains implementation of Reccurent Neural Networks using Keras and Tensorflow in order to make a stock trading bot. The inspiration for the project was SentDex's video about cryptocurrency trading bot using RNN which you can see [here](https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/).

## Table of Contents
- **Preprocessing data**
- **Spliting data**
- **Training models**
- **Testing models**
### Preprocessing data
First part of the project is loading and then preprocessing the data. I am using pandas dataframe for all data manipulation.
Big part of the preprocessing is adding indicators. If you can find data that already has indicator values built in, that would be great. Since I was using YahooFinance historical data, i needed to add some indicators by myself. They are: ATR, SMA, RSI. Here is the function for adding ATR, and others are pretty similar.
```
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
```
The idea was to use ATR as a Stoploss and Takeprofit, so the Risk to Reward Ration would be 1:1 all the time, and the SL and TP would be some coef*ATR. This is all implemented in ```df_1R(df)``` function, which appends 1 if the next move was to the upside and 0 for the downside.

After all the preprocessing, if we do ```print(df.tail())```, we end up with df that looks like this.
```
                 close       volume    ATR 14      SMA 14     RSI 14  target
time                                                                        
10/12/2020  123.239998   81312200.0  2.771501  120.154999  56.329987     1.0
11/12/2020  122.410004   86939800.0  2.857930  120.517143  56.267421     1.0
14/12/2020  121.779999   79184500.0  2.710787  121.083571  53.636485     1.0
15/12/2020  127.879997  157572300.0  2.915072  121.991428  61.915462     1.0
16/12/2020  127.809998   98208600.0  2.931501  122.832857  62.725394     1.0
```

### Spliting data
There are various ways to split data into train/validation/test, but most of them didn't work for me so I ended up writing several small functions to help me with that. The result was - 3 different sets(train/validation/test) where the ```testing_df``` was divided into 3 smaller tests(test1,test2,test3).
Here is the first part of the output:
```
Train data
               close       volume    ATR 14    SMA 14  RSI 14  target
15/12/1980  0.121652  175884800.0  0.007696  0.121652    50.0     1.0
16/12/1980  0.112723  105728000.0  0.008312  0.112723    50.0     1.0
17/12/1980  0.115513   86441600.0  0.006658  0.115513    50.0     1.0
18/12/1980  0.118862   73449600.0  0.005970  0.118862    50.0     1.0
19/12/1980  0.126116   48630400.0  0.006338  0.126116    50.0     1.0
Length:  8172 

Validation data
                close       volume    ATR 14     SMA 14     RSI 14  target
09/05/2013  16.313213  398487600.0  0.419255  15.519158  54.692189     0.0
10/05/2013  16.177500  334852000.0  0.413031  15.657678  51.078245     0.0
13/05/2013  16.240715  316948800.0  0.404587  15.781683  53.228353     0.0
14/05/2013  15.852143  447118000.0  0.379842  15.879643  46.431392     0.0
15/05/2013  15.316072  741613600.0  0.413056  15.931862  40.660749     1.0
Length:  908 

Test data
                close       volume    ATR 14     SMA 14     RSI 14  target
14/12/2016  28.797501  136127200.0  0.444000  27.938750  72.541905     1.0
15/12/2016  28.955000  186098000.0  0.455071  28.010714  73.577686     1.0
16/12/2016  28.992500  177404400.0  0.450964  28.089286  70.330984     1.0
19/12/2016  29.160000  111117600.0  0.445071  28.181786  72.524467     1.0
20/12/2016  29.237499   85700000.0  0.425964  28.296607  73.060689     1.0
Length:  1009 
```

### Training models
First part is the preprocessing function which scales all the data, because it is easier for the model to work with percentage change then real numbers. Also, you will not see any shuffle-ing in my preprocessing, or anywhere else because there is no point to do this with the financial data. There is small chance that price could bounce from 0.1$ to 100$ in one day, so I don't want to train the model for the crazy surcumstances. 
```SEQ_LEN``` is the number of candles before the current one that the model can use in order to make a prediction. With my experimenting, I found that smaller number of candles(rows in our df) leads to better results, although it is counterintuitive to me.
It is also the case with the ```BATCH_SIZE```, I had the best results when I used BATCH_SIZE of 8.
Rest of the code is about adding layers, then saving the best models and early stopping the training if needed.
Result of the training is the following:
```
train data: 8158 validation: 894
Epoch 1/10
1020/1020 [==============================] - 23s 13ms/step - loss: 0.8629 - accuracy: 0.5043 - val_loss: 0.7217 - val_accuracy: 0.5078

Epoch 00001: val_loss improved from inf to 0.72175, saving model to RNN_Final-01-0.508.hdf5
Epoch 2/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.7640 - accuracy: 0.5224 - val_loss: 0.7125 - val_accuracy: 0.5157

Epoch 00002: val_loss improved from 0.72175 to 0.71248, saving model to RNN_Final-02-0.516.hdf5
Epoch 3/10
1020/1020 [==============================] - 12s 11ms/step - loss: 0.7306 - accuracy: 0.5302 - val_loss: 0.6873 - val_accuracy: 0.5548

Epoch 00003: val_loss improved from 0.71248 to 0.68732, saving model to RNN_Final-03-0.555.hdf5
Epoch 4/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.7195 - accuracy: 0.5328 - val_loss: 0.7117 - val_accuracy: 0.5235

Epoch 00004: val_loss did not improve from 0.68732
Epoch 5/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.7117 - accuracy: 0.5344 - val_loss: 0.6914 - val_accuracy: 0.5503

Epoch 00005: val_loss did not improve from 0.68732
Epoch 6/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.7065 - accuracy: 0.5327 - val_loss: 0.6962 - val_accuracy: 0.5481

Epoch 00006: val_loss did not improve from 0.68732
Epoch 7/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.7055 - accuracy: 0.5331 - val_loss: 0.6981 - val_accuracy: 0.5369

Epoch 00007: val_loss did not improve from 0.68732
Epoch 8/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.6922 - accuracy: 0.5536 - val_loss: 0.6861 - val_accuracy: 0.5638

Epoch 00008: val_loss improved from 0.68732 to 0.68614, saving model to RNN_Final-08-0.564.hdf5
Epoch 9/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.6965 - accuracy: 0.5454 - val_loss: 0.6916 - val_accuracy: 0.5716

Epoch 00009: val_loss did not improve from 0.68614
Epoch 10/10
1020/1020 [==============================] - 12s 12ms/step - loss: 0.6885 - accuracy: 0.5621 - val_loss: 0.6852 - val_accuracy: 0.5772

Epoch 00010: val_loss improved from 0.68614 to 0.68519, saving model to RNN_Final-10-0.577.hdf5
```

### Testing models
After this, we can take our best model and test it on our testing data(test1,test2,test3)
The accuracy for these 3 tests is: ```60.25%, 56.21%, 67.70%``` which is pretty good.
The fact that accuracy is fluctuating by 5-10% easily is normal from the aspect of a trader, because in different market conditions the strategy will work differently, so we will get different gains.

### Connect with me
This wraps up my explanation of this project, I hope that I was clear and if you want to connect with me:
- Send me an email (lukasavic18@gmail.com)
- Follow me on [LinkedIn](https://www.linkedin.com/in/luka-savic-a73504206/)
