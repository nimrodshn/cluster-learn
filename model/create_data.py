#!/bin/env python3

import math
import pickle
import os
import pandas as pd
import numpy as np


TSLENGTH = 10
INPUT_LENGTH = 7
DATA_PATH = '../data/'
OUTPUT_PATH = '../'
COL_NAME = '\tCPU usage [%]'
MAX_DATA_FILES = 100


def create_data(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        col_name=COL_NAME,
        tslength=TSLENGTH,
        input_length=INPUT_LENGTH):
    """Create a time series dataset from the fastStorage dataset. (See README)
    """
    files = os.listdir(data_path)
    data_x = []
    data_y = []
    min_x = math.inf
    max_x = 0

    for filename in files[:MAX_DATA_FILES]:
        path = data_path + filename

        print("opening file:  " + path)
        dframe = pd.read_csv(path, sep=';', header=0)
        dframe = dframe[col_name]

        max_x = np.max([max_x, np.max(dframe)])
        min_x = np.min([min_x, np.min(dframe)])

        num_of_rows = dframe.shape[0]
        num_of_output_rows = num_of_rows - tslength

        for idx in range(num_of_output_rows):
            print("  %s [%05d:%05d]" % (filename, idx, num_of_output_rows))

            data_x.append(dframe[idx:idx + input_length])
            data_y.append(dframe[idx + input_length:idx + tslength])

    df_x = np.asarray(data_x)
    df_y = np.asarray(data_y)

    df_x = 2.0 * (df_x - min_x) / (max_x - min_x) - 1.0
    df_y = 2.0 * (df_y - min_x) / (max_x - min_x) - 1.0

    # Sanity check
    print("Input shape: %s" % str(df_x.shape))
    print("Label shape: %s" % str(df_y.shape))

    np.savetxt(output_path + 'raw_input.csv', df_x, delimiter=",")
    np.savetxt(output_path + 'raw_label.csv', df_y, delimiter=",")

    with open(output_path + "data.meta", "wb") as fd:
        pickle.dump([min_x, max_x], fd)


if __name__ == "__main__":
    create_data()
