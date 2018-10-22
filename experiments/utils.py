import argparse
from os import path
from top_level_file import base
import pandas as pd


def get_string_parser(*var_names):
    parser = argparse.ArgumentParser()

    for var_name in var_names:
        parser.add_argument(var_name)

    args = parser.parse_args()

    return args


def load_dataset(dataset_name):
    if dataset_name == 'fire':
        dataset = path.join(base, '..', "datasets", "cache", "fire_all.csv")
    else:
        dataset = path.join(base, '..', "datasets", "cache", "basketball_all.csv")

    return pd.read_csv(dataset)
