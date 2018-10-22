import pandas as pd


def load_data_frame(file_path):
    return pd.DataFrame.from_csv(file_path)


def get_x_y(dataframe):
    values = dataframe.values

    rows, cols = values.shape[0:2]

    return values[:, 1:(cols + 1)], values[:, 0:1]



