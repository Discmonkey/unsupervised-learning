import tensorflow 
from tensorflow import keras 
import pandas as pd 
from top_level_file import base
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import argparse


def main(dataset_name, reduction_name):
    df = pd.read_csv(os.path.join(base, "..", "datasets", "cache", "{}-train-{}.csv".format(dataset_name, reduction_name)))
    num_columns = len(df.columns) - 1 

    model = keras.models.Sequential()

    if dataset_name == 'basketball':
        model.add(keras.layers.InputLayer(input_shape=(num_columns,)))
        model.add(keras.layers.Dense(num_columns, activation='relu'))
        model.add(keras.layers.Dense(num_columns, activation='relu'))
        model.add(keras.layers.Dense(num_columns * 2, activation='relu'))
        model.add(keras.layers.Dense(30, activation='softmax'))
    else:
        model.add(keras.layers.InputLayer(input_shape=(num_columns,)))
        model.add(keras.layers.Dense(num_columns, activation='relu'))
        model.add(keras.layers.Dense(num_columns, activation='relu'))
        model.add(keras.layers.Dense(num_columns, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))

    # optimizer = keras.optimizers.SGD(lr=.0001, momentum=.01)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_dir = os.path.join(base, "training", dataset_name, reduction_name)
    saved_model_dir = os.path.join(base, "training", "fire", reduction_name + "saved_model")

    if os.path.exists(model_dir):
        os.removedirs(model_dir)

    os.makedirs(model_dir)

    x, y = df.values[:, 0:num_columns], df.values[:, num_columns:num_columns + 1]
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))

    new_y = enc.transform(y).toarray()

    model.fit(x, new_y, batch_size=32, epochs=10000, shuffle=True, validation_split=.1, callbacks=[
        keras.callbacks.TensorBoard(log_dir=model_dir, batch_size=32, write_graph=True),
        keras.callbacks.ModelCheckpoint(saved_model_dir, monitor='val_loss', save_best_only=True)
    ])


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("reduction")

    args = parser.parse_args()

    main(args.dataset, args.reduction)
