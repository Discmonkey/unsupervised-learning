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


def main(experiment_name): 
    df = pd.read_csv(os.path.join(base, "..", "datasets/cache/train_basketball.csv"))
    num_columns = len(df.columns) - 1 

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(num_columns,)))
    model.add(keras.layers.Dense(num_columns, activation='relu')) 
    model.add(keras.layers.Dense(num_columns, activation='relu'))
    model.add(keras.layers.Dense(num_columns * 2, activation='relu'))
    model.add(keras.layers.Dense(30, activation='softmax'))

    # optimizer = keras.optimizers.SGD(lr=.0001, momentum=.01)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_dir = os.path.join(base, "training", "fire", experiment_name)
    saved_model_dir = os.path.join(base, "training", "fire", experiment_name + "save_model")

    os.makedirs(model_dir)

    x, y = df.values[:, 1:(num_columns + 1)], df.values[:, 0:1]
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))

    new_y = enc.transform(y).toarray()

    model.fit(x, new_y, batch_size=32, epochs=10000, shuffle=True, validation_split=.1, callbacks=[
        keras.callbacks.TensorBoard(log_dir=model_dir, batch_size=32, write_graph=True),
        keras.callbacks.ModelCheckpoint(saved_model_dir, monitor='val_loss', save_best_only=True)
    ])


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name") 

    args = parser.parse_args()

    main(args.experiment_name) 
