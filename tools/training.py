import logging
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.optimizers import adam
from sklearn import preprocessing

from tools.dataCollectorHistoricalCSV import DataCollectorHistoricalCSV


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


class Training:
    SEQ_LEN = 60  # how long of a preceding sequence to collect for RNN
    FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict (hours)?
    EPOCHS = 10  # how many passes through our data
    BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

    def __init__(self):
        self.logger = logging.getLogger("rnnTrader.Training")
        self.logger.setLevel(logging.INFO)
        self.logger.info("Init Training")

    def run(self):
        self.logger.info("Start Training")
        df = self.build_data_frame()

        self.logger.info(df.head())

        times = sorted(df.index.values)
        last_5pct = sorted(df.index.values)[-int(0.05 * len(times))]

        validation_main_df = df[
            (df.index >= last_5pct)]
        df = df[(df.index < last_5pct)]

        train_x, train_y = self.preprocess_df(df)
        validation_x, validation_y = self.preprocess_df(validation_main_df)

        self.logger.info(f"train data: {len(train_x)} validation: {len(validation_x)}")
        self.logger.info(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
        self.logger.info(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

        self.start_training(train_x=train_x, train_y=train_y, validation_x=validation_x, validation_y=validation_y)

    def build_data_frame(self):
        self.logger.info("Build DataFrame")

        data_collect = DataCollectorHistoricalCSV()
        main_df = data_collect.load_data("Exmo_BTCUSD_1h")

        main_df.fillna(method="ffill", inplace=True)
        main_df.dropna(inplace=True)

        main_df['future'] = main_df['close'].shift(-self.FUTURE_PERIOD_PREDICT)
        main_df['target'] = list(map(classify, main_df['close'], main_df['future']))

        return main_df

    def preprocess_df(self, df):
        df = df.drop("future", 1)  # don't need this anymore.

        for col in df.columns:  # go through all of the columns
            if col != "target":  # normalize all ... except for the target itself!
                df[col] = df[
                    col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
                df.dropna(inplace=True)  # remove the nas created by pct_change
                df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

        df.dropna(inplace=True)  # cleanup again... jic.

        sequential_data = []  # this is a list that will CONTAIN the sequences
        prev_days = deque(
            maxlen=self.SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

        for i in df.values:  # iterate over the values
            prev_days.append([n for n in i[:-1]])  # store all but the target
            if len(prev_days) == self.SEQ_LEN:  # make sure we have 60 sequences!
                sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

        random.shuffle(sequential_data)  # shuffle for good measure.

        buys = []  # list that will store our buy sequences and targets
        sells = []  # list that will store our sell sequences and targets

        for seq, target in sequential_data:  # iterate over the sequential data
            if target == 0:  # if it's a "not buy"
                sells.append([seq, target])  # append to sells list
            elif target == 1:  # otherwise if the target is a 1...
                buys.append([seq, target])  # it's a buy!

        random.shuffle(buys)  # shuffle the buys
        random.shuffle(sells)  # shuffle the sells!

        lower = min(len(buys), len(sells))  # what's the shorter length?

        buys = buys[:lower]  # make sure both lists are only up to the shortest length.
        sells = sells[:lower]  # make sure both lists are only up to the shortest length.

        sequential_data = buys + sells  # add them together
        random.shuffle(
            sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

        X = []
        y = []

        for seq, target in sequential_data:  # going over our new sequential data
            X.append(seq)  # X is the sequences
            y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

        return np.array(X), y  # return X and y...and make X a numpy array!

    def start_training(self, train_x, train_y, validation_x, validation_y):
        model = Sequential()
        model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Set Optimizer
        opt = adam(lr=0.001, decay=1e-6)

        # Compile model
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        tensorboard = TensorBoard(log_dir="logs/{}".format(self.NAME))

        filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
        checkpoint = ModelCheckpoint(
            "models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max'))  # saves only the best ones

        # Train model
        history = model.fit(
            train_x, train_y,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint],
        )

        # Score model
        score = model.evaluate(validation_x, validation_y, verbose=0)
        self.logger.info('Test loss:', score[0])
        self.logger.info('Test accuracy:', score[1])
        # Save model
        model.save("models/{}".format(self.NAME))
